use hnsw_itu::{Distance, Graph, Index, IndexBuilder, IndexVis, NSW, Point};
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

pub struct ThesisIndexBuilder {
    options: ThesisIndexOptions,
}

impl ThesisIndexBuilder {
    pub fn new(options: ThesisIndexOptions) -> Self {
        Self { options }
    }

    pub fn build<P: Point + Clone + Send + Sync>(
        self,
        queries: &Vec<P>,
        data: Vec<P>,
    ) -> ThesisIndex<P> {
        let query_graph = self.build_nsw(&queries.iter().collect());
        let estimated_gt = self.estimate_gt(&query_graph, &data.iter().collect());

        let mut counter = HashMap::new();
        for knn in estimated_gt.iter() {
            for &Distance { key: p, .. } in knn.iter() {
                *counter.entry(p).or_insert(0) += 1;
            }
        }

        let mut values = counter.values().collect::<Vec<_>>();
        values.sort_unstable();
        values.reverse();
        info!("Top 5 most common point counts");
        for i in values.iter().take(5) {
            info!("{:?}", i);
        }

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

        // Construct bipartite graph
        info!("Constructing bipartite graph...");
        let mut bipartite_graph =
            AdjListGraph::with_nodes_additional(data.iter().collect(), queries.len());
        for (q, closest) in estimated_gt.iter().enumerate() {
            //if let Some(&Distance { key: p, .. }) = closest.first() {
            //    bipartite_graph.add_directed_edge(p, q + query_count)
            //}
            for &Distance { key: p, .. } in closest.iter().take(self.options.e) {
                bipartite_graph.add_edge(p, q + queries.len())
            }
            for &Distance { key: p, .. } in closest.iter().skip(self.options.e) {
                bipartite_graph.add_directed_edge(q + queries.len(), p);
            }
        }

        let mut lists = bipartite_graph
            .adj_lists
            .iter()
            .take(data.len())
            .cloned()
            .collect::<Vec<_>>();
        lists.sort_unstable_by_key(|l| l.len());
        lists.reverse();
        let empty_cnt = lists.iter().filter(|l| l.is_empty()).count();
        let non_empty_cnt = lists.iter().filter(|l| !l.is_empty()).count();
        info!("empty: {empty_cnt}, not empty: {non_empty_cnt}");
        info!("Top 5 most dense neighborhoods");
        for l in lists.iter().take(5) {
            info!("{:?}", l.len());
        }

        // Bipartite projection
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

    fn connectivity_enhancement<'a, P: Point + Send + Sync>(
        &self,
        data: &Vec<P>,
        projected_graph: &mut AdjListGraph<&'a P>,
        entry: usize,
    ) {
        info!("Finding connectivity candidates...");
        let all_candidates: Vec<(usize, MinMaxHeap<Distance<'_, &P>>)> = data
            .par_iter()
            .map(|p| projected_graph.search(&p, entry, self.options.m))
            .enumerate()
            .collect();

        info!("Enhancing connectivity...");
        let conn_graph = projected_graph.clone();
        let nodes = &conn_graph.nodes;
        let adj_locks: Vec<RwLock<HashSet<usize>>> = conn_graph
            .adj_lists
            .into_iter()
            .map(|s| RwLock::new(s))
            .collect();

        all_candidates.into_par_iter().for_each(|(i, candidates)| {
            let selected_neighbors = self.select_neighbors(candidates);
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
                let p_neighbors: Vec<usize> = self
                    .select_neighbors(p_candidates)
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

    fn build_nsw<'a, P: Point + Send + Sync>(&self, data: &Vec<&'a P>) -> NSW<&'a P> {
        info!("Constructing NSW graph...");
        let mut graph_builder = NSWBuilder::new(NSWOptions {
            ef_construction: 96,
            connections: 32,
            max_connections: 96,
            size: data.len(),
        });
        graph_builder.extend_parallel(data.to_vec());
        graph_builder.build()
    }

    fn estimate_gt<'a, P: Point + Send + Sync>(
        &self,
        query_graph: &NSW<&'a P>,
        data: &Vec<&P>,
    ) -> Vec<Vec<Distance<'a, P>>> {
        let mut estimated_gt: Vec<RwLock<Vec<Distance<P>>>> = vec![];
        for _ in 0..query_graph.graph().size() {
            estimated_gt.push(RwLock::new(vec![]));
        }

        info!("Finding k-nearest query neighbors...");
        data.par_iter().enumerate().for_each(|(i, d)| {
            let mut vis = HashSet::with_capacity(256);
            let knn = if self.options.vis {
                query_graph.search_vis(d, self.options.qk, self.options.qef, &mut vis)
            } else {
                query_graph.search(d, self.options.qk, self.options.qef)
            };

            for Distance {
                key,
                distance,
                point: _point,
            } in knn.into_iter().chain(vis.into_iter())
            {
                let mut l = estimated_gt[key].write().unwrap();
                l.push(Distance::new(
                    distance,
                    i,
                    query_graph.graph().nodes()[key],
                ));

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
        if candidates.len() <= max {
            return candidates.drain_asc().take(max).collect();
        }

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
