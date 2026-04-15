use crate::labels::{
    FilteredSearchOptions, LabelSet, filtered_search_locked, find_medoids, intersection_is_subset,
};
use crate::vamana::index::FilteredVamana;
use hnsw_itu::{Distance, Index, IndexBuilder, MinK, Point};
use min_max_heap::MinMaxHeap;
use rand::seq::{IndexedRandom, SliceRandom};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::info;

pub struct FilteredVamanaOptions {
    pub alpha: f32,
    /// Search list size
    pub l: usize,
    /// Degree bound
    pub r: usize,
    /// Medoid sample size
    pub threshold: usize,
    pub labels: Vec<LabelSet>,
}

impl std::fmt::Debug for FilteredVamanaOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteredVamanaOptions")
            .field("alpha", &self.alpha)
            .field("l", &self.l)
            .field("r", &self.r)
            .field("threshold", &self.threshold)
            .field("labels", &format_args!("<{} items>", self.labels.len()))
            .finish()
    }
}

pub struct FilteredVamanaBuilder<P> {
    options: FilteredVamanaOptions,
    index: FilteredVamana<P>,
}

impl<P> FilteredVamanaBuilder<P> {
    pub fn new(mut options: FilteredVamanaOptions) -> Self {
        let mut index = FilteredVamana::default();
        let mut labels = Vec::new();
        std::mem::swap(&mut labels, &mut options.labels);
        index.labels = labels;

        Self { options, index }
    }
}

impl<P: Point + Send + Sync> IndexBuilder<P> for FilteredVamanaBuilder<P> {
    type Index = FilteredVamana<P>;

    fn add(&mut self, point: P) {
        self.index.graph.add(point);
    }

    fn build(mut self) -> Self::Index {
        let n = self.index.size();
        self.index.start_nodes = find_medoids(&self.index.labels, self.options.threshold);
        info!(
            "Building graph for {n} points (threads: {})...",
            rayon::current_num_threads()
        );

        let nodes = &self.index.graph.nodes;
        let labels = &self.index.labels;
        let start_nodes = &self.index.start_nodes;
        let alpha = self.options.alpha;
        let r = self.options.r;
        let l = self.options.l;

        let adj_locks: Vec<RwLock<HashSet<usize>>> = self
            .index
            .graph
            .adj_lists
            .drain(..)
            .map(RwLock::new)
            .collect();

        info!("Initializing random {r}-regular graph...");
        {
            let mut rng = rand::rng();
            let all_ids: Vec<usize> = (0..n).collect();
            for i in 0..n {
                let mut neighbors = HashSet::with_capacity(r);
                for &s in all_ids.sample(&mut rng, r + 1) {
                    if s != i && neighbors.len() < r {
                        neighbors.insert(s);
                    }
                }
                *adj_locks[i].write().unwrap() = neighbors;
            }
        }

        let progress = AtomicUsize::new(0);
        let log_interval = (n / 5).max(1);

        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rand::rng());

        order.into_par_iter().for_each(|i| {
            let done = progress.fetch_add(1, Ordering::Relaxed);
            if done.is_multiple_of(log_interval) {
                info!("{done}/{n} ({:.1}%)", done as f64 * 100.0 / n as f64);
            }

            let point = &nodes[i];
            let point_labels = &labels[i];

            let search_start: Vec<usize> = point_labels
                .iter()
                .filter_map(|f| start_nodes.get(&f).copied())
                .collect();

            if search_start.is_empty() {
                return;
            }

            let search_options = FilteredSearchOptions {
                ef: l,
                start_nodes: &search_start,
                labels: point_labels,
            };
            let mut vis = HashSet::with_capacity(2048);
            let _ = filtered_search_locked(
                point,
                0,
                nodes,
                &adj_locks,
                labels,
                &search_options,
                &mut vis,
            );

            let mut prune_candidates: MinMaxHeap<_> = vis.into_iter().min_k(l).into();

            {
                let current_neighbors: Vec<usize> =
                    adj_locks[i].read().unwrap().iter().copied().collect();
                prune_candidates.extend(current_neighbors.into_iter().map(|nb| {
                    let np = &nodes[nb];
                    Distance::new(np.distance(point), nb, np)
                }));
            }

            let neighbor_keys =
                robust_prune_inline(prune_candidates, point_labels, labels, alpha, r);

            {
                let mut adj_i = adj_locks[i].write().unwrap();
                adj_i.clear();
                adj_i.extend(neighbor_keys.iter().copied());
            }

            for &j in &neighbor_keys {
                // Insert reverse edge and check degree in a single write lock
                let needs_prune = {
                    let mut adj_j = adj_locks[j].write().unwrap();
                    adj_j.insert(i);
                    adj_j.len() > r
                };

                if needs_prune {
                    let j_point = &nodes[j];
                    let j_labels = &labels[j];
                    let j_neighbors: Vec<usize> =
                        adj_locks[j].read().unwrap().iter().copied().collect();

                    let j_candidates: MinMaxHeap<_> = j_neighbors
                        .into_iter()
                        .map(|k| {
                            let kp = &nodes[k];
                            Distance::new(j_point.distance(kp), k, kp)
                        })
                        .collect();

                    let j_new_neighbors =
                        robust_prune_inline(j_candidates, j_labels, labels, alpha, r);

                    {
                        let mut adj_j = adj_locks[j].write().unwrap();
                        adj_j.clear();
                        adj_j.extend(j_new_neighbors);
                    }
                }
            }
        });

        self.index.graph.adj_lists = adj_locks
            .into_iter()
            .map(|lock| lock.into_inner().unwrap())
            .collect();

        self.index
    }
}

fn robust_prune_inline<P: Point>(
    candidates: MinMaxHeap<Distance<'_, P>>,
    point_labels: &LabelSet,
    labels: &[LabelSet],
    alpha: f32,
    max_degree: usize,
) -> Vec<usize> {
    let sorted = candidates.into_vec_asc();

    let len = sorted.len();
    let mut pruned = vec![false; len];
    let mut result = Vec::with_capacity(max_degree);

    for i in 0..len {
        if pruned[i] {
            continue;
        }

        let cand = &sorted[i];
        let cand_labels = &labels[cand.key];

        result.push(cand.key);
        if result.len() >= max_degree {
            break;
        }

        for j in (i + 1)..len {
            if pruned[j] {
                continue;
            }
            let other = &sorted[j];
            let other_labels = &labels[other.key];
            if !intersection_is_subset(other_labels, point_labels, cand_labels) {
                continue;
            }
            if alpha * cand.point.distance(other.point) <= other.distance {
                pruned[j] = true;
            }
        }
    }

    result
}

impl<P: Point + Send + Sync> Extend<P> for FilteredVamanaBuilder<P> {
    fn extend<T: IntoIterator<Item = P>>(&mut self, iter: T) {
        for i in iter {
            self.add(i);
        }
    }
}
