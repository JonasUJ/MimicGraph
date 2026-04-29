use crate::gt::*;
use hnsw_itu::{Distance, Index, Point};
use min_max_heap::MinMaxHeap;
use rayon::prelude::*;
use roargraph::{AdjListGraph, select_neighbors, select_neighbors_max};
use std::collections::HashSet;
use std::sync::RwLock;
use tracing::info;

pub mod filtered;
pub mod plain;

#[derive(Debug, Clone)]
pub struct MimicGraphOptions {
    /// Out-degree bound
    pub m: usize,
    /// Candidate pool size
    pub l: usize,
    /// KNN pruned set size
    pub p: usize,
    /// Bipartite reverse edges
    pub e: usize,
    /// Query graph k
    pub qk: usize,
    /// Query graph ef
    pub qef: usize,
    /// Connectivity enhancement
    pub con: bool,
    /// Use visited nodes as candidates
    pub vis: bool,
    /// Percentage of corpus used as build queries (0, 100]
    pub q: f32,
}

impl Default for MimicGraphOptions {
    fn default() -> Self {
        Self {
            m: 32,
            l: 300,
            p: 100,
            e: 32,
            qk: 0,
            qef: 100,
            con: false,
            vis: true,
            q: 10.0,
        }
    }
}

impl MimicGraphOptions {
    pub fn tuned<P>(data: &[P], queries: &[P]) -> MimicGraphOptions
    where
        P: Point + Sync,
    {
        info!("Auto tuning options...");

        let k = 100;
        let slots = 10;
        let data_size = data.len().min(data.len().clamp(10000, 50_000));
        let queries_size = queries
            .len()
            .min(((data_size * slots) as f32 / k as f32).round() as usize);

        let data = &data[..data_size];
        let queries = &queries[..queries_size];

        let gt = compute_ground_truth(queries, data, k);

        let (data_fraction, data_spread) = fraction_and_spread(&gt, data_size, k);
        info!(data_fraction, data_spread, data_size, queries_size);

        let options = MimicGraphOptions::from_fraction_and_spread(data_fraction, data_spread);
        info!(
            ?options.m, ?options.l, ?options.p,
            "Tuned options",
        );

        options
    }

    pub fn from_fraction_and_spread(data_fraction: f32, data_spread: f32) -> MimicGraphOptions {
        Self {
            m: (100.0 - 100.0 * (0.3 * data_fraction + 0.7 * data_spread)).clamp(16.0, 80.0)
                as usize,
            l: (650.0 * (0.7 * data_fraction + 0.3 * data_spread)).clamp(50.0, 600.0) as usize,
            p: (300.0 * (0.7 * data_fraction + 0.3 * data_spread)).clamp(100.0, 300.0) as usize,
            ..Default::default()
        }
    }
}

pub trait Builder<P: Point + Send + Sync> {
    type Index: Index<P>;
    type QueryGraph<'a>: Index<&'a P>
    where
        P: 'a;

    fn options(&self) -> &MimicGraphOptions;

    fn build(self, queries: &[P], data: Vec<P>) -> Self::Index;

    fn build_query_graph<'a>(&self, data: &[&'a P]) -> Self::QueryGraph<'a>;

    fn estimate_gt<'a>(
        &self,
        query_graph: &Self::QueryGraph<'a>,
        data: &[&'a P],
    ) -> Vec<Vec<Distance<'a, P>>>;
}

trait BuilderExt<P: Point + Send + Sync> {
    fn bipartite_projection<'a>(
        &self,
        data: Vec<&'a P>,
        query_count: usize,
        estimated_gt: Vec<Vec<Distance<P>>>,
    ) -> AdjListGraph<&'a P>;

    fn connectivity_enhancement(
        &self,
        data: &[P],
        projected_graph: &mut AdjListGraph<&P>,
        entry: usize,
    );

    fn neighborhood_aware_projection(
        &self,
        bipartite_graph: AdjListGraph<&P>,
        projected_graph: &mut AdjListGraph<&P>,
    );
}

impl<B, P> BuilderExt<P> for B
where
    B: Builder<P> + Sync,
    P: Point + Send + Sync,
{
    fn bipartite_projection<'a>(
        &self,
        data: Vec<&'a P>,
        query_count: usize,
        estimated_gt: Vec<Vec<Distance<P>>>,
    ) -> AdjListGraph<&'a P> {
        let data_count = data.len();
        let mut bipartite_graph = AdjListGraph::with_nodes_additional(data, query_count);
        for (q, closest) in estimated_gt.iter().enumerate() {
            for &Distance { key: p, .. } in closest.iter().take(self.options().e) {
                bipartite_graph.add_edge(p, q + data_count)
            }
            for &Distance { key: p, .. } in closest.iter().skip(self.options().e) {
                bipartite_graph.add_directed_edge(q + data_count, p);
            }
        }

        bipartite_graph
    }

    fn connectivity_enhancement(
        &self,
        data: &[P],
        projected_graph: &mut AdjListGraph<&P>,
        entry: usize,
    ) {
        info!("Finding connectivity candidates...");
        let all_candidates = data
            .par_iter()
            .map(|p| projected_graph.search(&p, self.options().m, &entry))
            .enumerate()
            .collect::<Vec<_>>();

        info!("Enhancing connectivity...");
        let conn_graph = projected_graph.clone();
        let nodes = &conn_graph.nodes;
        let adj_locks: Vec<RwLock<HashSet<usize>>> =
            conn_graph.adj_lists.into_iter().map(RwLock::new).collect();

        all_candidates.into_par_iter().for_each(|(i, candidates)| {
            let selected_neighbors = select_neighbors(candidates.into(), self.options().m);
            {
                let mut adj_i = adj_locks[i].write().unwrap();
                adj_i.clear();
                adj_i.extend(selected_neighbors.iter().map(|d| d.key));
            }

            for p in &selected_neighbors {
                let current_neighbors: Vec<usize> =
                    { adj_locks[p.key].read().unwrap().iter().copied().collect() };
                let p_candidates = MinMaxHeap::from_iter(
                    current_neighbors
                        .into_iter()
                        .chain(std::iter::once(i))
                        .map(|n| {
                            let neighbor_point = &nodes[n];
                            Distance::new(p.point.distance(neighbor_point), n, neighbor_point)
                        }),
                );
                let p_neighbors: Vec<usize> = select_neighbors(p_candidates, self.options().m)
                    .into_iter()
                    .map(|d| d.key)
                    .collect();
                {
                    let mut adj_p = adj_locks[p.key].write().unwrap();
                    adj_p.clear();
                    adj_p.extend(p_neighbors);
                }
            }
        });

        let conn_adj_lists: Vec<HashSet<usize>> = adj_locks
            .into_iter()
            .map(|l| l.into_inner().unwrap())
            .collect();

        for (i, list) in conn_adj_lists.iter().enumerate() {
            let mut final_neighbors = projected_graph.neighborhood(i).collect::<HashSet<usize>>();
            final_neighbors.extend(list.iter().copied());
            projected_graph.set_neighbors(i, final_neighbors.into_iter());
        }
    }

    fn neighborhood_aware_projection(
        &self,
        bipartite_graph: AdjListGraph<&P>,
        projected_graph: &mut AdjListGraph<&P>,
    ) {
        let nodes = &projected_graph.nodes;
        let adj_locks: Vec<RwLock<HashSet<usize>>> = projected_graph
            .adj_lists
            .drain(..)
            .map(RwLock::new)
            .collect();

        (0..nodes.len()).into_par_iter().for_each(|x| {
            let mut out_neighbors = bipartite_graph.neighborhood(x).peekable();
            if out_neighbors.peek().is_none() {
                return;
            }

            let x_point = &nodes[x];
            let mut candidates = MinMaxHeap::new();
            'outer: for s in out_neighbors {
                for neighbor in bipartite_graph.neighborhood(s) {
                    if neighbor != x {
                        let neighbor_point = &nodes[neighbor];
                        candidates.push(Distance::new(
                            x_point.distance(neighbor_point),
                            neighbor,
                            neighbor_point,
                        ));

                        if candidates.len() >= self.options().l {
                            break 'outer;
                        }
                    }
                }
            }

            let selected_neighbors = select_neighbors_max(candidates, self.options().m);
            let selected_neighbors = selected_neighbors.iter().map(|d| d.key).collect::<Vec<_>>();

            {
                let mut adj_x = adj_locks[x].write().unwrap();
                adj_x.clear();
                adj_x.extend(selected_neighbors.iter().copied());
            }

            for &p in &selected_neighbors {
                let point = &nodes[p];
                let current_neighbors: Vec<usize> =
                    { adj_locks[p].read().unwrap().iter().copied().collect() };
                let new_candidates = MinMaxHeap::from_iter(
                    current_neighbors
                        .into_iter()
                        .chain(std::iter::once(x))
                        .map(|n| {
                            let neighbor_point = &nodes[n];
                            Distance::new(point.distance(neighbor_point), n, neighbor_point)
                        }),
                );
                let selected = select_neighbors_max(new_candidates, self.options().m)
                    .iter()
                    .map(|d| d.key)
                    .collect::<Vec<_>>();
                {
                    let mut adj_p = adj_locks[p].write().unwrap();
                    adj_p.clear();
                    adj_p.extend(selected);
                }
            }
        });

        projected_graph.adj_lists = adj_locks
            .into_iter()
            .map(|l| l.into_inner().unwrap())
            .collect();
    }
}
