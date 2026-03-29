use crate::vamana::LabelledPoint;
use crate::vamana::filtered::FilteredVamanaOptions;
use hnsw_itu::{Distance, Index, IndexVis, Point};
use min_max_heap::MinMaxHeap;
use roargraph::AdjListGraph;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

pub struct FilteredVamana<P> {
    pub(crate) graph: AdjListGraph<P>,
}

impl<P> FilteredVamana<P> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<P> Default for FilteredVamana<P> {
    fn default() -> Self {
        Self {
            graph: AdjListGraph::new(),
        }
    }
}

#[derive(Debug)]
pub struct FilteredVamanaSearchOptions {
    pub l: usize,
    /// Start nodes
    pub s: Vec<usize>,
}

impl<P: Point> Index<LabelledPoint<P>> for FilteredVamana<LabelledPoint<P>> {
    type Options = FilteredVamanaSearchOptions;
    type Query = LabelledPoint<P>;

    fn size(&self) -> usize {
        self.graph.size()
    }

    fn search(
        &'_ self,
        query: &Self::Query,
        k: usize,
        options: &Self::Options,
    ) -> Vec<Distance<'_, LabelledPoint<P>>> {
        let mut visited = HashSet::with_capacity(2048);
        self.search_vis(query, k, options, &mut visited)
    }
}

impl<P: Point> IndexVis<LabelledPoint<P>> for FilteredVamana<LabelledPoint<P>> {
    fn search_vis<'a>(
        &'a self,
        query: &Self::Query,
        k: usize,
        options: &Self::Options,
        vis: &mut HashSet<Distance<'a, LabelledPoint<P>>>,
    ) -> Vec<Distance<'a, LabelledPoint<P>>> {
        let mut candidates = MinMaxHeap::with_capacity(options.l);

        for &s in options.s.iter() {
            let sp = self.graph.get(s).unwrap();
            if sp.labels.intersection(&query.labels).next().is_some() {
                candidates.push(Distance::new(sp.distance(query), s, sp));
            }
        }

        while let Some(candidate) = candidates.pop_min() {
            if vis.contains(&candidate) {
                continue;
            }
            vis.insert(candidate.clone());

            candidates.extend(self.graph.neighborhood(candidate.key).filter_map(|n| {
                let np = self.graph.get(n).unwrap();
                if vis.contains(&Distance::new(0.0, n, np)) {
                    return None;
                }

                if query.labels.intersection(&np.labels).next().is_some() {
                    return Some(Distance::new(np.distance(query), n, np));
                }

                None
            }));

            while candidates.len() >= options.l {
                candidates.pop_max();
            }
        }

        candidates.drain_asc().take(k).collect()
    }
}
