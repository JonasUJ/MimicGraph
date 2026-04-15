use hnsw_itu::{Distance, Point};
use min_max_heap::MinMaxHeap;
use rand::seq::IteratorRandom;
use roargraph::AdjListGraph;
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

pub type LabelSet = hi_sparse_bitset::BitSet<hi_sparse_bitset::config::_128bit>;

/// Returns true if two label sets share at least one element.
#[inline]
pub fn labels_intersect(a: &LabelSet, b: &LabelSet) -> bool {
    !(a & b).is_empty()
}

/// Returns true if (a ∩ b) ⊆ c.
#[inline]
pub fn intersection_is_subset(a: &LabelSet, b: &LabelSet, c: &LabelSet) -> bool {
    ((a & b) - c).is_empty()
}

pub fn find_medoids(labels: &[LabelSet], threshold: usize) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    let mut filters = HashSet::new();
    for ls in labels {
        filters.extend(ls.iter());
    }
    let mut counter = vec![0; labels.len()];

    let mut label_map = HashMap::new();

    for (i, ls) in labels.iter().enumerate() {
        for l in ls.iter() {
            label_map.entry(l).or_insert(vec![]).push(i);
        }
    }

    let mut rng = rand::rng();
    for f in filters {
        let sample = label_map
            .get(&f)
            .unwrap()
            .iter()
            .sample(&mut rng, threshold);
        let (entry, _) = sample
            .iter()
            .map(|&i| (i, counter[*i]))
            .min_by_key(|(_, c)| *c)
            .unwrap();
        *counter.get_mut(*entry).unwrap() += 1;
        map.insert(f, *entry);
    }

    map
}

pub struct FilteredSearchOptions<'a> {
    pub ef: usize,
    pub start_nodes: &'a [usize],
    pub labels: &'a LabelSet,
}

impl std::fmt::Debug for FilteredSearchOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteredSearchOptions")
            .field("ef", &self.ef)
            .field("start_nodes", &self.start_nodes)
            .finish()
    }
}

fn filtered_search_core<'a, P: Point>(
    query: &P,
    k: usize,
    labels: &[LabelSet],
    options: &FilteredSearchOptions<'_>,
    vis: &mut HashSet<Distance<'a, P>>,
    get_point: impl Fn(usize) -> &'a P,
    get_neighbors: impl Fn(usize) -> Vec<usize>,
) -> Vec<Distance<'a, P>> {
    let mut candidates = MinMaxHeap::with_capacity(options.ef);
    let mut results = MinMaxHeap::with_capacity(options.ef);
    let mut vis_keys: HashSet<usize> = vis.iter().map(|d| d.key).collect();

    for &s in options.start_nodes.iter() {
        if vis_keys.contains(&s) {
            continue;
        }
        if !labels_intersect(&labels[s], options.labels) {
            continue;
        }
        let s_point = get_point(s);
        let dist = Distance::new(s_point.distance(query), s, s_point);
        vis_keys.insert(s);
        vis.insert(dist.clone());
        candidates.push(dist);
    }

    while let Some(candidate) = candidates.pop_min() {
        results.push(candidate.clone());

        if results.len() > options.ef {
            results.pop_max();
        }

        if results.len() >= options.ef
            && let Some(worst) = results.peek_max()
            && candidate.distance > worst.distance
        {
            break;
        }

        for n in get_neighbors(candidate.key) {
            if vis_keys.contains(&n) {
                continue;
            }
            if !labels_intersect(options.labels, &labels[n]) {
                continue;
            }
            let n_point = get_point(n);
            let dist = Distance::new(n_point.distance(query), n, n_point);
            vis_keys.insert(n);
            vis.insert(dist.clone());
            candidates.push(dist);
        }

        while candidates.len() >= options.ef {
            candidates.pop_max();
        }
    }

    results.drain_asc().take(k).collect()
}

pub trait FilteredGraphExt<P: Point> {
    fn filtered_search<'a>(
        &'a self,
        query: &P,
        k: usize,
        labels: &[LabelSet],
        options: &FilteredSearchOptions<'_>,
    ) -> Vec<Distance<'a, P>>;

    fn filtered_search_vis<'a>(
        &'a self,
        query: &P,
        k: usize,
        labels: &[LabelSet],
        options: &FilteredSearchOptions<'_>,
        vis: &mut HashSet<Distance<'a, P>>,
    ) -> Vec<Distance<'a, P>>;
}

impl<P: Point> FilteredGraphExt<P> for AdjListGraph<P> {
    fn filtered_search<'a>(
        &'a self,
        query: &P,
        k: usize,
        labels: &[LabelSet],
        options: &FilteredSearchOptions<'_>,
    ) -> Vec<Distance<'a, P>> {
        let mut visited = HashSet::with_capacity(2048);
        self.filtered_search_vis(query, k, labels, options, &mut visited)
    }

    fn filtered_search_vis<'a>(
        &'a self,
        query: &P,
        k: usize,
        labels: &[LabelSet],
        options: &FilteredSearchOptions<'_>,
        vis: &mut HashSet<Distance<'a, P>>,
    ) -> Vec<Distance<'a, P>> {
        filtered_search_core(
            query,
            k,
            labels,
            options,
            vis,
            |i| self.get(i).unwrap(),
            |i| self.neighborhood(i).collect(),
        )
    }
}

/// Filtered greedy search over RwLock-wrapped adjacency lists.
/// Same algorithm as `FilteredGraphExt::filtered_search_vis`, but reads
/// neighbors through `RwLock`s so it can run concurrently with writers.
pub fn filtered_search_locked<'a, P: Point>(
    query: &P,
    k: usize,
    nodes: &'a [P],
    adj_locks: &[RwLock<HashSet<usize>>],
    labels: &[LabelSet],
    options: &FilteredSearchOptions<'_>,
    vis: &mut HashSet<Distance<'a, P>>,
) -> Vec<Distance<'a, P>> {
    filtered_search_core(
        query,
        k,
        labels,
        options,
        vis,
        |i| &nodes[i],
        |i| adj_locks[i].read().unwrap().iter().copied().collect(),
    )
}
