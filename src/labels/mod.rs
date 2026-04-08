use hnsw_itu::{Distance, Point};
use min_max_heap::MinMaxHeap;
use rand::seq::IteratorRandom;
use roargraph::AdjListGraph;
use std::collections::{HashMap, HashSet};

pub fn find_medoids<P: Point>(
    data: &[P],
    labels: &HashMap<usize, HashSet<usize>>,
    threshold: usize,
) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    let filters = labels.keys().copied().collect::<HashSet<_>>();
    let mut counter: HashMap<usize, usize> = HashMap::from_iter(filters.iter().map(|&i| (i, 0)));

    let mut label_map = HashMap::new();

    for (i, point, labels) in data
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p, labels.get(&i).unwrap()))
    {
        for label in labels.iter() {
            label_map.entry(label).or_insert(vec![]).push(i);
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
            .map(|&i| (i, counter.get(&f).unwrap()))
            .min_by_key(|(_, c)| *c)
            .unwrap();
        *counter.get_mut(&f).unwrap() += 1;
        map.insert(f, *entry);
    }

    map
}

#[derive(Debug)]
pub struct FilteredSearchOptions<'a> {
    pub ef: usize,
    pub start_nodes: &'a [usize],
    pub labels: &'a HashSet<usize>,
}

pub trait FilteredGraphExt<P: Point> {
    fn filtered_search<'a>(
        &'a self,
        query: &P,
        k: usize,
        labels: &HashMap<usize, HashSet<usize>>,
        options: &FilteredSearchOptions<'_>,
    ) -> Vec<Distance<'a, P>>;

    fn filtered_search_vis<'a>(
        &'a self,
        query: &P,
        k: usize,
        labels: &HashMap<usize, HashSet<usize>>,
        options: &FilteredSearchOptions<'_>,
        vis: &mut HashSet<Distance<'a, P>>,
    ) -> Vec<Distance<'a, P>>;
}

impl<P: Point> FilteredGraphExt<P> for AdjListGraph<P> {
    fn filtered_search<'a>(
        &'a self,
        query: &P,
        k: usize,
        labels: &HashMap<usize, HashSet<usize>>,
        options: &FilteredSearchOptions<'_>,
    ) -> Vec<Distance<'a, P>> {
        let mut visited = HashSet::with_capacity(2048);
        self.filtered_search_vis(query, k, labels, options, &mut visited)
    }

    fn filtered_search_vis<'a>(
        &'a self,
        query: &P,
        k: usize,
        labels: &HashMap<usize, HashSet<usize>>,
        options: &FilteredSearchOptions<'_>,
        vis: &mut HashSet<Distance<'a, P>>,
    ) -> Vec<Distance<'a, P>> {
        let mut candidates = MinMaxHeap::with_capacity(options.ef);

        for &s in options.start_nodes.iter() {
            let s_point = self.get(s).unwrap();
            let s_labels = labels.get(&s).unwrap();
            if s_labels.intersection(&options.labels).next().is_some() {
                candidates.push(Distance::new(s_point.distance(query), s, s_point));
            }
        }

        while let Some(candidate) = candidates.pop_min() {
            if vis.contains(&candidate) {
                continue;
            }
            vis.insert(candidate.clone());

            candidates.extend(self.neighborhood(candidate.key).filter_map(|n| {
                let n_point = self.get(n).unwrap();
                let n_labels = labels.get(&n).unwrap();
                if vis.contains(&Distance::new(0.0, n, n_point)) {
                    return None;
                }

                if options.labels.intersection(&n_labels).next().is_some() {
                    return Some(Distance::new(n_point.distance(query), n, n_point));
                }

                None
            }));

            while candidates.len() >= options.ef {
                candidates.pop_max();
            }
        }

        candidates.drain_asc().take(k).collect()
    }
}
