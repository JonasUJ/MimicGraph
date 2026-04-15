use crate::labels::{LabelSet, find_medoids};
use crate::thesis_index::{Builder, BuilderExt, ThesisIndexOptions};
use crate::vamana::filtered::{FilteredVamanaBuilder, FilteredVamanaOptions};
use crate::vamana::index::{FilteredVamana, FilteredVamanaSearchOptions};
use hnsw_itu::{Distance, Index, IndexBuilder, IndexVis, Point};
use rayon::prelude::*;
use roargraph::AdjListGraph;
use std::collections::HashSet;
use std::sync::RwLock;
use tracing::{info, warn};

pub type FilteredThesisIndex<P> = FilteredVamana<P>;

pub struct FilteredThesisIndexOptions {
    pub base_options: ThesisIndexOptions,
    pub threshold: usize,
    pub labels: Vec<LabelSet>,
    pub query_labels: Vec<LabelSet>,
}

impl std::fmt::Debug for FilteredThesisIndexOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteredThesisIndexOptions")
            .field("base_options", &self.base_options)
            .field("threshold", &self.threshold)
            .field("labels", &format_args!("<{} items>", self.labels.len()))
            .field(
                "query_labels",
                &format_args!("<{} items>", self.query_labels.len()),
            )
            .finish()
    }
}

pub struct FilteredThesisIndexBuilder {
    options: FilteredThesisIndexOptions,
}

impl FilteredThesisIndexBuilder {
    pub fn new(options: FilteredThesisIndexOptions) -> Self {
        Self { options }
    }
}

impl<P: Point + Send + Sync> Builder<P> for FilteredThesisIndexBuilder {
    type Index = FilteredThesisIndex<P>;
    type QueryGraph<'a>
        = FilteredVamana<&'a P>
    where
        P: 'a;

    fn options(&self) -> &ThesisIndexOptions {
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
            //self.connectivity_enhancement(&data, &mut projected_graph, entry);
        }

        let (_, adj_lists) = projected_graph.consume();

        info!("FilteredThesisIndex construction complete");
        FilteredThesisIndex {
            start_nodes,
            labels: self.options.labels,
            graph: AdjListGraph {
                nodes: data,
                adj_lists,
            },
        }
    }

    fn build_query_graph<'a>(&self, data: &[&'a P]) -> Self::QueryGraph<'a> {
        info!("Constructing Query graph...");
        let mut graph_builder = FilteredVamanaBuilder::new(FilteredVamanaOptions {
            alpha: 1.2,
            l: 90,
            r: 96,
            threshold: 1000,
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
