use crate::thesis_index::{Builder, BuilderExt, ThesisIndexOptions};
use hnsw_itu::{Distance, Index, IndexBuilder, IndexVis, NSW, Point};
use hnsw_itu::{NSWBuilder, NSWOptions};
use rayon::prelude::*;
use roargraph::AdjListGraph;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;
use tracing::info;

#[derive(Serialize, Deserialize)]
pub struct ThesisIndex<T> {
    pub entry: usize,
    pub(crate) graph: AdjListGraph<T>,
}

impl<P: Point> Index<P> for ThesisIndex<P> {
    type Options<'a> = usize;

    fn size(&self) -> usize {
        self.graph.size()
    }

    fn search(&'_ self, query: &P, k: usize, ef: &Self::Options<'_>) -> Vec<Distance<'_, P>> {
        let mut res = self.graph.search(query, *ef, &self.entry);
        res.truncate(k);
        res
    }
}

pub struct ThesisIndexBuilder {
    options: ThesisIndexOptions,
}

impl ThesisIndexBuilder {
    pub fn new(options: ThesisIndexOptions) -> Self {
        Self { options }
    }
}

impl<P: Point + Send + Sync> Builder<P> for ThesisIndexBuilder {
    type Index = ThesisIndex<P>;
    type QueryGraph<'a>
        = NSW<&'a P>
    where
        P: 'a;

    fn options(&self) -> &ThesisIndexOptions {
        &self.options
    }

    fn build(self, queries: &[P], data: Vec<P>) -> Self::Index {
        let query_graph = self.build_query_graph(&queries.iter().collect::<Vec<_>>());
        let estimated_gt = self.estimate_gt(&query_graph, &data.iter().collect::<Vec<_>>());

        info!("Computing most common nearest neighbor...");
        let mut counter = HashMap::new();
        for knn in estimated_gt.iter() {
            if let Some(&Distance { key: p, .. }) = knn.first() {
                *counter.entry(p).or_insert(0) += 1;
            }
        }

        let entry = *counter
            .iter()
            .max_by_key(|(_, v)| **v)
            .map(|(k, _)| k)
            .expect("no points");

        info!("Constructing bipartite graph...");
        let bipartite_graph =
            self.bipartite_projection(data.iter().collect(), queries.len(), estimated_gt);

        info!("Projecting bipartite graph...");
        let mut projected_graph = AdjListGraph::with_nodes(data.iter().collect());
        self.neighborhood_aware_projection(bipartite_graph, &mut projected_graph);

        if self.options.con {
            self.connectivity_enhancement(&data, &mut projected_graph, entry);
        }

        let (_, adj_lists) = projected_graph.consume();

        info!("ThesisIndex construction complete");
        ThesisIndex {
            entry,
            graph: AdjListGraph {
                nodes: data,
                adj_lists,
            },
        }
    }

    fn build_query_graph<'a>(&self, data: &[&'a P]) -> Self::QueryGraph<'a> {
        info!("Constructing Query graph...");
        let mut graph_builder = NSWBuilder::new(NSWOptions {
            ef_construction: 500,
            connections: 32,
            max_connections: 96,
            size: data.len(),
        });
        graph_builder.extend_parallel(data.to_vec());

        graph_builder.build()
    }

    fn estimate_gt<'a>(
        &self,
        query_graph: &Self::QueryGraph<'a>,
        data: &[&'a P],
    ) -> Vec<Vec<Distance<'a, P>>> {
        let mut estimated_gt: Vec<RwLock<Vec<Distance<P>>>> = vec![];
        for _ in 0..query_graph.size() {
            estimated_gt.push(RwLock::new(vec![]));
        }

        info!("Finding k-nearest query neighbors...");
        data.par_iter().enumerate().for_each(|(i, d)| {
            let mut vis = HashSet::with_capacity(256);
            let knn = if self.options.vis {
                query_graph.search_vis(d, self.options.qk, &self.options.qef, &mut vis)
            } else {
                query_graph.search(d, self.options.qk, &self.options.qef)
            };

            for Distance {
                key,
                distance,
                point: _point,
            } in knn.into_iter().chain(vis.into_iter())
            {
                let mut l = estimated_gt[key].write().unwrap();
                l.push(Distance::new(distance, i, data[i]));

                if l.len() > self.options.p * 2 {
                    l.sort();
                    l.truncate(self.options.p);
                }
            }
        });

        info!("Pruning k-nearest query neighbors...");
        estimated_gt
            .into_par_iter()
            .map(|knn| {
                let mut knn = knn.into_inner().unwrap();
                knn.sort();
                knn.truncate(self.options.p);
                knn
            })
            .collect()
    }
}
