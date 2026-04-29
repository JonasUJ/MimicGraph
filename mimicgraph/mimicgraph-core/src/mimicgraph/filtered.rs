use crate::bitset::Bitset;
use crate::labels::{LabelSet, find_medoids};
use crate::mimicgraph::{Builder, BuilderExt, MimicGraphOptions};
use crate::vamana::FilteredMimicGraphOptions;
use crate::vamana::filtered::{FilteredVamanaBuilder, FilteredVamanaOptions};
use crate::vamana::index::{FilteredVamana, FilteredVamanaSearchOptions};
use hnsw_itu::{Distance, Index, IndexBuilder, IndexVis, Point};
use min_max_heap::MinMaxHeap;
use rayon::prelude::*;
use roargraph::AdjListGraph;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{Mutex, RwLock};
use tracing::{info, warn};

#[derive(Serialize, Deserialize)]
pub struct FilteredMimicGraph<P> {
    pub inner: FilteredVamana<P>,
    /// For each label, a bitset of point indices that carry that label.
    pub inverted_index: Vec<Bitset>,
}

pub struct FilteredMimicGraphSearchOptions<'a> {
    pub ef: usize,
    pub labels: &'a LabelSet,
    /// Linear scan searches labels with <= scan_limit points.
    /// 0 disables scanning.
    pub scan_limit: usize,
}

impl std::fmt::Debug for FilteredMimicGraphSearchOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteredMimicGraphSearchOptions")
            .field("ef", &self.ef)
            .field("scan_limit", &self.scan_limit)
            .finish()
    }
}

impl<P: Point> Index<P> for FilteredMimicGraph<P> {
    type Options<'a> = FilteredMimicGraphSearchOptions<'a>;

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn search(&'_ self, query: &P, k: usize, options: &Self::Options<'_>) -> Vec<Distance<'_, P>> {
        let mut visited = HashSet::with_capacity(2048);
        self.search_vis(query, k, options, &mut visited)
    }
}

impl<P: Point> IndexVis<P> for FilteredMimicGraph<P> {
    fn search_vis<'a>(
        &'a self,
        query: &P,
        k: usize,
        options: &Self::Options<'_>,
        vis: &mut HashSet<Distance<'a, P>>,
    ) -> Vec<Distance<'a, P>> {
        let vamana_options = FilteredVamanaSearchOptions {
            ef: options.ef,
            labels: options.labels,
        };

        if options.scan_limit == 0 {
            return self.inner.search_vis(query, k, &vamana_options, vis);
        }

        let mut scan_labels = LabelSet::new();

        // Collect scannable labels
        for label in options.labels.iter() {
            let count = self.inverted_index.get(label).map_or(0, |b| b.count());
            if count > 0 && count <= options.scan_limit {
                scan_labels.set(label);
            }
        }

        if scan_labels.is_empty() {
            // All labels are above scan_limit
            return self.inner.search_vis(query, k, &vamana_options, vis);
        }

        // Linear scan for the scannable labels
        let nodes = self.inner.graph.nodes();
        let mut results = MinMaxHeap::with_capacity(k + 1);
        let mut seen = Bitset::new();

        // Union all scannable posting lists so each point is visited once
        let mut scan_points = LabelSet::new();
        for label in scan_labels.iter() {
            if let Some(points) = self.inverted_index.get(label) {
                scan_points |= points;
            }
        }

        for point_idx in scan_points.iter() {
            let point = &nodes[point_idx];
            let dist = Distance::new(point.distance(query), point_idx, point);
            seen.set(point_idx);
            results.push(dist);
            if results.len() > k {
                results.pop_max();
            }
        }

        let search_labels = vamana_options.labels - scan_labels;

        if search_labels.is_empty() {
            // All query labels were scannable
            return results.drain_asc().collect();
        }

        // Search for labels via graph
        // We're searching for all query labels, even the ones we scanned.
        let search_results = self.inner.search_vis(query, k, &vamana_options, vis);

        // Merge scan and search results, skipping duplicates
        for d in search_results {
            if seen.set(d.key) {
                results.push(d);
            }
        }

        results.drain_asc().take(k).collect()
    }
}

pub struct FilteredMimicGraphBuilder {
    options: FilteredMimicGraphOptions,
}

impl FilteredMimicGraphBuilder {
    pub fn new(options: FilteredMimicGraphOptions) -> Self {
        Self { options }
    }
}

impl<P: Point + Send + Sync> Builder<P> for FilteredMimicGraphBuilder {
    type Index = FilteredMimicGraph<P>;
    type QueryGraph<'a>
        = FilteredVamana<&'a P>
    where
        P: 'a;

    fn options(&self) -> &MimicGraphOptions {
        &self.options.base_options
    }

    fn build(self, queries: &[P], data: Vec<P>) -> Self::Index {
        let query_graph = self.build_query_graph(&queries.iter().collect::<Vec<_>>());
        let estimated_gt = self.estimate_gt(&query_graph, &data.iter().collect::<Vec<_>>());

        info!("Finding start nodes...");
        let start_nodes = find_medoids(&self.options.labels, self.options.threshold);

        info!("Constructing bipartite graph...");
        let bipartite_graph =
            self.bipartite_projection(data.iter().collect(), queries.len(), estimated_gt);

        info!("Projecting bipartite graph...");
        let mut projected_graph = AdjListGraph::with_nodes(data.iter().collect());
        self.neighborhood_aware_projection(bipartite_graph, &mut projected_graph);

        if self.options.base_options.con {
            warn!("Connectivity enhancement is not implemented for filtered index");
        }

        let (_, adj_lists) = projected_graph.consume();

        info!("Building inverted index...");
        let num_labels = self
            .options
            .labels
            .iter()
            .flat_map(|ls| ls.iter())
            .max()
            .map_or(0, |m| m + 1);
        let inverted_index: Vec<Mutex<Bitset>> =
            (0..num_labels).map(|_| Mutex::new(Bitset::new())).collect();
        self.options
            .labels
            .par_iter()
            .enumerate()
            .for_each(|(point_idx, label_set)| {
                for label in label_set.iter() {
                    inverted_index[label].lock().unwrap().set(point_idx);
                }
            });
        let inverted_index: Vec<Bitset> = inverted_index
            .into_iter()
            .map(|m| m.into_inner().unwrap())
            .collect();

        info!("FilteredMimicGraph construction complete");
        FilteredMimicGraph {
            inner: FilteredVamana {
                start_nodes,
                labels: self.options.labels,
                graph: AdjListGraph {
                    nodes: data,
                    adj_lists,
                },
            },
            inverted_index,
        }
    }

    fn build_query_graph<'a>(&self, data: &[&'a P]) -> Self::QueryGraph<'a> {
        info!("Constructing Query graph...");
        let mut graph_builder = FilteredVamanaBuilder::new(FilteredVamanaOptions {
            alpha: 1.4,
            l: 90,
            r: 96,
            threshold: data.len() / 100,
            labels: self.options.query_labels.clone(),
        });
        graph_builder.extend(data.to_vec());

        graph_builder.build()
    }

    fn estimate_gt<'a>(
        &self,
        query_graph: &Self::QueryGraph<'a>,
        data: &[&'a P],
    ) -> Vec<Vec<Distance<'a, P>>> {
        let options = &self.options.base_options;

        let mut estimated_gt: Vec<RwLock<Vec<Distance<P>>>> = vec![];
        for _ in 0..query_graph.size() {
            estimated_gt.push(RwLock::new(vec![]));
        }

        info!("Finding k-nearest query neighbors...");
        data.par_iter().enumerate().for_each(|(i, d)| {
            let d_labels = &self.options.labels[i];

            let mut vis = HashSet::with_capacity(256);
            let search_options = FilteredVamanaSearchOptions {
                ef: options.qef,
                labels: d_labels,
            };

            let knn = if options.vis {
                query_graph.search_vis(d, options.qk, &search_options, &mut vis)
            } else {
                query_graph.search(d, options.qk, &search_options)
            };

            for Distance {
                key,
                distance,
                point: _point,
            } in knn.into_iter().chain(vis.into_iter())
            {
                let mut l = estimated_gt[key].write().unwrap();
                l.push(Distance::new(distance, i, data[i]));

                if l.len() > options.p * 2 {
                    l.sort();
                    l.truncate(options.p);
                }
            }
        });

        info!("Pruning k-nearest query neighbors...");
        estimated_gt
            .into_par_iter()
            .map(|knn| {
                let mut knn = knn.into_inner().unwrap();
                knn.sort();
                knn.truncate(options.p);
                knn
            })
            .collect()
    }
}
