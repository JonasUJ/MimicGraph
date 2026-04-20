pub(crate) use crate::labels::{FilteredGraphExt, FilteredSearchOptions, LabelSet};
use hnsw_itu::{Distance, Index, IndexVis, Point};
use roargraph::AdjListGraph;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Serialize, Deserialize)]
pub struct FilteredVamana<P> {
    pub(crate) start_nodes: HashMap<usize, usize>,
    pub(crate) graph: AdjListGraph<P>,
    pub(crate) labels: Vec<LabelSet>,
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
            labels: Vec::new(),
        }
    }
}

pub struct FilteredVamanaSearchOptions<'a> {
    pub ef: usize,
    pub labels: &'a LabelSet,
}

impl std::fmt::Debug for FilteredVamanaSearchOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteredVamanaSearchOptions")
            .field("ef", &self.ef)
            .finish()
    }
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
        let search_start: Vec<usize> = options
            .labels
            .iter()
            .filter_map(|f| self.start_nodes.get(&f).copied())
            .collect();

        if search_start.is_empty() {
            return vec![];
        }

        let search_options = FilteredSearchOptions {
            ef: options.ef,
            start_nodes: &search_start,
            labels: options.labels,
        };

        self.graph
            .filtered_search_vis(query, k, &self.labels, &search_options, vis)
    }
}
