use crate::gt::compute_ground_truth;
use crate::mimicgraph::{Builder, BuilderExt, MimicGraphOptions};
use hnsw_itu::{Distance, Index, IndexBuilder, IndexVis, NSW, Point};
use hnsw_itu::{NSWBuilder, NSWOptions};
use rayon::prelude::*;
use roargraph::AdjListGraph;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;
use tracing::info;

#[derive(Serialize, Deserialize)]
pub struct MimicGraph<T> {
    pub entry: usize,
    pub graph: AdjListGraph<T>,
}

impl<P: Point> Index<P> for MimicGraph<P> {
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

pub struct MimicGraphBuilder {
    options: MimicGraphOptions,
}

impl MimicGraphBuilder {
    pub fn new(options: MimicGraphOptions) -> Self {
        Self { options }
    }
}

impl<P: Point + Send + Sync> Builder<P> for MimicGraphBuilder {
    type Index = MimicGraph<P>;
    type QueryGraph<'a>
        = NSW<&'a P>
    where
        P: 'a;

    fn options(&self) -> &MimicGraphOptions {
        &self.options
    }

    fn build(mut self, queries: &[P], data: Vec<P>) -> Self::Index {
        //self.options = self.auto_tune(&data, queries);

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

        info!("MimicGraph construction complete");
        MimicGraph {
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

impl MimicGraphBuilder {
    fn auto_tune<P>(&self, data: &[P], queries: &[P]) -> MimicGraphOptions
    where
        P: Point + Sync,
    {
        info!("Auto tuning options...");

        let mut options = self.options.clone();

        let k = 100;
        let slots = 10;
        let data_size = data.len().min(data.len().clamp(10000, 100_000));
        let queries_size = queries
            .len()
            .min(((data_size * slots) as f32 / k as f32).round() as usize);

        let data = &data[..data_size];
        let queries = &queries[..queries_size];

        let gt = compute_ground_truth(queries, data, k);

        let mut counts = vec![0usize; data_size];

        for knn in gt.iter() {
            for (i, _) in knn {
                counts[*i] += 1;
            }
        }

        let nonzero_count = counts.iter().filter(|&&i| i != 0).count();

        // Fraction of data points that appear in the ground truth at least once
        let data_fraction = nonzero_count as f32 / data_size as f32;

        let data_spread = {
            let total = counts.iter().sum::<usize>() as f32;
            let sum_sq: f32 = counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f32 / total;
                    p * p
                })
                .sum();

            let effective_support = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };
            let min_support = k.min(data_size) as f32;
            let max_support = data_size as f32;

            ((effective_support - min_support) / (max_support - min_support)).clamp(0.0, 1.0)
        };

        info!(data_fraction, data_spread, data_size, queries_size);

        options.m = (40.0 * data_fraction + 2.0 / data_spread).clamp(12.0, 64.0) as usize;
        options.e = (options.m as f32 / (2.0 + 2.0 * data_fraction)).max(1.0) as usize;
        options.l = (500.0 * data_fraction * data_spread).max(50.0) as usize;
        options.p = options.l.min(200);
        options.qef = (100.0 - 50.0 * data_fraction * data_spread).round() as usize;

        info!(
            "Tuned options: m={}, e={}, l={}, p={}, qef={}",
            options.m, options.e, options.l, options.p, options.qef
        );

        options
    }
}
