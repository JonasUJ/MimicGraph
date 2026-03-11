use hnsw_itu::{Distance, Index, IndexBuilder, Point};
use hnsw_itu::{NSWBuilder, NSWOptions};
use min_max_heap::MinMaxHeap;
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

pub trait Searchable<P: Point> {
    fn search(&'_ self, query: &P, entry: usize, ef: usize) -> MinMaxHeap<Distance<'_, P>>;
}

impl<P: Point> Searchable<P> for ThesisIndex<P> {
    fn search(&'_ self, query: &P, entry: usize, ef: usize) -> MinMaxHeap<Distance<'_, P>> {
        self.graph.search(query, entry, ef)
    }
}

impl<P: Point> Searchable<P> for AdjListGraph<P> {
    fn search(&'_ self, query: &P, entry: usize, ef: usize) -> MinMaxHeap<Distance<'_, P>> {
        let medoid_element = self.get(entry).expect("entry point was not in graph");
        let query_distance = Distance::new(medoid_element.distance(query), entry, medoid_element);

        let mut visited = HashSet::with_capacity(2048);
        visited.insert(entry);
        let mut w = MinMaxHeap::from_iter([query_distance.clone()]);
        let mut candidates = MinMaxHeap::from_iter([query_distance]);

        while !candidates.is_empty() {
            let c = candidates.pop_min().expect("candidates can't be empty");
            let f = w.peek_max().expect("w can't be empty");

            if c.distance > f.distance {
                break;
            }

            for e in self.neighborhood(c.key) {
                if visited.contains(&e) {
                    continue;
                }

                visited.insert(e);
                let f = w.peek_max().expect("w can't be empty");

                let point = self.get(e).unwrap();
                let e_dist = Distance::new(point.distance(query), e, point);

                if e_dist.distance >= f.distance && w.len() >= ef {
                    continue;
                }

                candidates.push(e_dist.clone());
                w.push(e_dist);

                if w.len() > ef {
                    w.pop_max();
                }
            }
        }

        w
    }
}

#[derive(Debug, Clone)]
pub struct ThesisIndexOptions {
    /// Out-degree bound
    pub(crate) m: usize,
    /// Candidate pool size
    pub(crate) l: usize,
    /// KNN pruned set size
    pub(crate) p: usize,
}

pub struct ThesisIndexBuilder {
    options: ThesisIndexOptions,
}

impl ThesisIndexBuilder {
    pub fn new(options: ThesisIndexOptions) -> Self {
        Self { options }
    }

    pub fn build<P: Point + Clone + Send + Sync>(
        self,
        queries: Vec<P>,
        data: Vec<P>,
    ) -> ThesisIndex<P> {
        info!("Constructing NSW graph...");
        let query_count = queries.len();
        let mut query_graph_builder = NSWBuilder::new(NSWOptions {
            ef_construction: 100,
            connections: 16,
            max_connections: 32,
            size: query_count,
        });
        query_graph_builder.extend_parallel(queries);
        let query_graph = query_graph_builder.build();

        // TODO: Make k and ef parameters
        info!("Finding k-nearest query neighbors...");
        let knns = query_graph.knns(&data, 100, 200);

        info!("Pruning k-nearest query neighbors...");
        let mut knns_inverted = vec![vec![]; query_count];
        for (i, knn) in knns.iter().enumerate() {
            for Distance {
                key,
                distance,
                point,
            } in knn
            {
                knns_inverted[*key].push(Distance::new(*distance, i, *point));
            }
        }

        knns_inverted.par_iter_mut().for_each(|knn| {
            *knn =
                self.select_neighbors_max(knn.drain(..).collect::<MinMaxHeap<_>>(), self.options.p);
        });

        info!("Computing most common nearest neighbor...");
        let mut counter = HashMap::new();
        for knn in knns_inverted.iter() {
            if let Some(&Distance { key: p, .. }) = knn.first() {
                *counter.entry(p).or_insert(0) += 1;
            }
        }

        let entry = *counter
            .iter()
            .max_by_key(|(_, v)| **v)
            .map(|(k, _)| k)
            .expect("no points");

        // Construct bipartite graph
        info!("Constructing bipartite graph...");
        let mut bipartite_graph =
            AdjListGraph::with_nodes_additional(data.iter().collect(), query_count);
        for (q, closest) in knns_inverted.iter().enumerate() {
            if let Some(&Distance { key: p, .. }) = closest.first() {
                bipartite_graph.add_directed_edge(p, q + query_count)
            }
            for &Distance { key: p, .. } in closest.iter().skip(1) {
                bipartite_graph.add_directed_edge(q + query_count, p);
            }
        }

        // Bipartite projection
        info!("Projecting bipartite graph...");
        let mut projected_graph = AdjListGraph::with_nodes(data.iter().collect());
        self.neighborhood_aware_projection(bipartite_graph, &mut projected_graph);

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

    // Heuristic
    #[allow(unused)]
    fn select_neighbors<'a, P: Point>(
        &self,
        mut candidates: MinMaxHeap<Distance<'a, P>>,
    ) -> Vec<Distance<'a, P>> {
        let mut return_list = Vec::<Distance<'a, P>>::new();

        while let Some(e) = candidates.pop_min() {
            if return_list.len() >= self.options.m {
                break;
            }

            if return_list
                .iter()
                .all(|r| e.point.distance(r.point) > e.distance)
            {
                return_list.push(e);
            }
        }

        return_list
    }

    fn select_neighbors_max<'a, P: Point>(
        &self,
        mut candidates: MinMaxHeap<Distance<'a, P>>,
        max: usize,
    ) -> Vec<Distance<'a, P>> {
        let mut return_list = Vec::<Distance<'a, P>>::new();
        let mut rejects = MinMaxHeap::new();

        while let Some(e) = candidates.pop_min() {
            if return_list.len() >= max {
                break;
            }

            if return_list
                .iter()
                .all(|r| e.point.distance(r.point) > e.distance)
            {
                return_list.push(e);
            } else {
                rejects.push(e);
            }
        }

        return_list.extend(rejects.drain_asc().take(max - return_list.len()));
        return_list
    }

    fn neighborhood_aware_projection<P: Point + Send + Sync>(
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

                        if candidates.len() >= self.options.l {
                            break 'outer;
                        }
                    }
                }
            }

            let selected_neighbors = self.select_neighbors_max(candidates, self.options.m);
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
                let selected = self
                    .select_neighbors_max(new_candidates, self.options.m)
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
