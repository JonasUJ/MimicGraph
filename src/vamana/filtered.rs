use crate::vamana::index::{FilteredVamana, FilteredVamanaSearchOptions};
use crate::vamana::{GraphExt, LabelledPoint, find_medoids};
use hnsw_itu::{Distance, Index, IndexBuilder, IndexVis, Point};
use min_max_heap::MinMaxHeap;
use roargraph::AdjListGraph;
use std::collections::{HashMap, HashSet};

pub struct FilteredVamanaOptions {
    pub filters: HashSet<usize>,
    pub alpha: f32,
    /// Search list size
    pub l: usize,
    /// Degree bound
    pub r: usize,
    /// Medoid sample size
    pub threshold: usize,
}

impl Default for FilteredVamanaOptions {
    fn default() -> Self {
        Self {
            filters: HashSet::default(),
            alpha: 1.2,
            l: 90,
            r: 96,
            threshold: 0,
        }
    }
}

impl FilteredVamanaOptions {
    pub fn with_filters(mut self, filters: HashSet<usize>) -> Self {
        self.filters = filters;
        self
    }
}

pub struct FilteredVamanaBuilder<P> {
    options: FilteredVamanaOptions,
    index: FilteredVamana<P>,
}

impl<P> FilteredVamanaBuilder<P> {
    pub fn new(options: FilteredVamanaOptions) -> Self {
        Self {
            options,
            index: FilteredVamana::default(),
        }
    }
}

impl<P> Default for FilteredVamanaBuilder<P> {
    fn default() -> Self {
        Self::new(FilteredVamanaOptions::default())
    }
}

impl<P: Point> FilteredVamanaBuilder<LabelledPoint<P>> {
    fn pruned_neighbor_keys<'a>(
        &'a self,
        node: usize,
        mut candidates: MinMaxHeap<Distance<'a, LabelledPoint<P>>>,
    ) -> Vec<usize> {
        self.index
            .graph
            .robust_prune(node, &mut candidates, self.options.alpha, self.options.r)
            .iter()
            .map(|d| d.key)
            .collect()
    }
}

impl<P: Point> IndexBuilder<LabelledPoint<P>> for FilteredVamanaBuilder<LabelledPoint<P>> {
    type Index = FilteredVamana<LabelledPoint<P>>;

    fn add(&mut self, point: LabelledPoint<P>) {
        self.index.graph.add(point);
    }

    fn build(mut self) -> Self::Index {
        let start_nodes = find_medoids(
            &self.index.graph.nodes,
            &self.options.filters,
            self.options.threshold,
        );

        for i in 0..self.index.size() {
            let point = self.index.graph.get(i).unwrap();
            let search_start = point.labels.iter().map(|f| start_nodes[f]).collect();

            let search_options = FilteredVamanaSearchOptions {
                l: self.options.l,
                s: search_start,
            };
            let mut vis = HashSet::with_capacity(2048);
            let _ = self.index.search_vis(point, 0, &search_options, &mut vis);

            let candidates = vis.into_iter().collect();
            let neighbors = self.pruned_neighbor_keys(i, candidates);
            self.index.graph.set_neighbors(i, neighbors.iter().copied());

            for j in neighbors {
                self.index.graph.add_directed_edge(j, i);
                if let Some(other_neighbors) = self.index.graph.adj_lists.get(j)
                    && other_neighbors.len() > self.options.r
                {
                    let other = self.index.graph.get(j).unwrap();
                    let candidates = other_neighbors
                        .iter()
                        .map(|&k| {
                            let p = self.index.graph.get(k).unwrap();
                            Distance::new(other.distance(p), k, p)
                        })
                        .collect();
                    let neighbors = self.pruned_neighbor_keys(j, candidates);
                    self.index.graph.set_neighbors(j, neighbors.iter().copied());
                }
            }
        }

        self.index
    }
}

impl<P: Point> Extend<LabelledPoint<P>> for FilteredVamanaBuilder<LabelledPoint<P>> {
    fn extend<T: IntoIterator<Item = LabelledPoint<P>>>(&mut self, iter: T) {
        for i in iter {
            self.add(i);
        }
    }
}
