pub(crate) use crate::labels::{FilteredGraphExt, FilteredSearchOptions};
use hnsw_itu::{Distance, Index, IndexVis, Point};
use roargraph::AdjListGraph;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

pub struct FilteredVamana<P> {
    pub(crate) start_nodes: HashMap<usize, usize>,
    pub(crate) graph: AdjListGraph<P>,
    pub(crate) labels: HashMap<usize, HashSet<usize>>,
}

impl<P> FilteredVamana<P> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<P> Default for FilteredVamana<P> {
    fn default() -> Self {
        Self {
            start_nodes: HashMap::new(),
            graph: AdjListGraph::new(),
            labels: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct FilteredVamanaSearchOptions<'a> {
    pub ef: usize,
    pub labels: &'a HashSet<usize>,
}

impl<P: Point> Index<P> for FilteredVamana<P> {
    type Options<'a> = FilteredVamanaSearchOptions<'a>;

    fn size(&self) -> usize {
        self.graph.size()
    }

    fn search(&'_ self, query: &P, k: usize, options: &Self::Options<'_>) -> Vec<Distance<'_, P>> {
        let mut visited = HashSet::with_capacity(2048);
        self.search_vis(query, k, options, &mut visited)
    }
}

impl<P: Point> IndexVis<P> for FilteredVamana<P> {
    fn search_vis<'a>(
        &'a self,
        query: &P,
        k: usize,
        options: &Self::Options<'_>,
        vis: &mut HashSet<Distance<'a, P>>,
    ) -> Vec<Distance<'a, P>> {
        let search_start = options
            .labels
            .iter()
            .map(|f| self.start_nodes[f])
            .collect::<Vec<_>>();

        let search_options = FilteredSearchOptions {
            ef: options.ef,
            start_nodes: &search_start,
            labels: options.labels,
        };

        self.graph
            .filtered_search_vis(query, k, &self.labels, &search_options, vis)
    }
}
