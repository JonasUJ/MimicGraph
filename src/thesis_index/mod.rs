use hnsw_itu::{Distance, Graph, HNSW, HNSWBuilder, Index, IndexBuilder, IndexVis, NSW, Point};
use min_max_heap::MinMaxHeap;
use rayon::prelude::*;
use roargraph::{AdjListGraph, select_neighbors, select_neighbors_max};
use std::collections::{HashSet};
use std::sync::RwLock;
use tracing::info;

pub mod filtered;
pub mod plain;

#[derive(Debug, Clone)]
pub struct ThesisIndexOptions {
    /// Out-degree bound
    pub(crate) m: usize,
    /// Candidate pool size
    pub(crate) l: usize,
    /// KNN pruned set size
    pub(crate) p: usize,
    /// Bipartite reverse edges
    pub(crate) e: usize,
    /// Query graph k
    pub(crate) qk: usize,
    /// Query graph ef
    pub(crate) qef: usize,
    /// Connectivity enhancement
    pub(crate) con: bool,
    /// Use visited nodes as candidates
    pub(crate) vis: bool,
}

pub trait Builder<P: Point + Send + Sync> {
    type Index: Index<P>;
    type QueryGraph<'a>: Index<&'a P>
    where
        P: 'a;

    fn options(&self) -> &ThesisIndexOptions;

    fn build(self, queries: &Vec<P>, data: Vec<P>) -> Self::Index;

    fn build_query_graph<'a>(&self, data: &Vec<&'a P>) -> Self::QueryGraph<'a>;

    fn estimate_gt<'a>(
        &self,
        query_graph: &Self::QueryGraph<'a>,
        data: &Vec<&'a P>,
    ) -> Vec<Vec<Distance<'a, P>>>;
}

trait BuilderExt<P: Point + Send + Sync> {
    fn bipartite_projection<'a>(
        &self,
        data: Vec<&'a P>,
        query_count: usize,
        estimated_gt: Vec<Vec<Distance<P>>>,
    ) -> AdjListGraph<&'a P>;

    fn connectivity_enhancement<'a>(
        &self,
        data: &Vec<P>,
        projected_graph: &mut AdjListGraph<&'a P>,
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

    fn connectivity_enhancement<'a>(
        &self,
        data: &Vec<P>,
        projected_graph: &mut AdjListGraph<&'a P>,
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
        let adj_locks: Vec<RwLock<HashSet<usize>>> = conn_graph
            .adj_lists
            .into_iter()
            .map(|s| RwLock::new(s))
            .collect();

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

        (0..projected_graph.nodes().len())
            .into_iter()
            .for_each(|i| {
                let mut final_neighbors =
                    projected_graph.neighborhood(i).collect::<HashSet<usize>>();
                final_neighbors.extend(conn_adj_lists[i].iter().copied());
                projected_graph.set_neighbors(i, final_neighbors.into_iter());
            });
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
            .map(|s| RwLock::new(s))
            .collect();

        (0..nodes.len()).into_par_iter().for_each(|x| {
            let mut out_neighbors = bipartite_graph.neighborhood(x).into_iter().peekable();
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
