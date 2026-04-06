use hnsw_itu::{Distance, Point};
use min_max_heap::MinMaxHeap;
use rand::seq::IteratorRandom;
use roargraph::AdjListGraph;
use std::collections::{HashMap, HashSet};

pub mod filtered;
pub mod index;
pub mod stitched;

#[derive(Debug)]
pub struct LabelledPoint<P> {
    pub point: P,
    pub labels: HashSet<usize>,
}

impl<P: Point> Point for LabelledPoint<P> {
    fn distance(&self, other: &Self) -> f32 {
        self.point.distance(&other.point)
    }
}

fn find_medoids<P: Point>(
    data: &[LabelledPoint<P>],
    filters: &HashSet<usize>,
    threshold: usize,
) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    let mut counter: HashMap<usize, usize> = HashMap::from_iter(filters.iter().map(|&i| (i, 0)));

    let mut label_map = HashMap::new();

    for (i, p) in data.iter().enumerate() {
        for label in p.labels.iter() {
            label_map.entry(label).or_insert(vec![]).push(i);
        }
    }

    let mut rng = rand::rng();
    for &f in filters {
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

trait GraphExt<P> {
    fn robust_prune<'a>(
        &'a self,
        point: usize,
        candidates: &mut MinMaxHeap<Distance<'a, LabelledPoint<P>>>,
        alpha: f32,
        max_degree: usize,
    ) -> Vec<Distance<'a, LabelledPoint<P>>>;
}

impl<P: Point> GraphExt<P> for AdjListGraph<LabelledPoint<P>> {
    fn robust_prune<'a>(
        &'a self,
        point: usize,
        candidates: &mut MinMaxHeap<Distance<'a, LabelledPoint<P>>>,
        alpha: f32,
        max_degree: usize,
    ) -> Vec<Distance<'a, LabelledPoint<P>>> {
        let p = self.get(point).expect("point not in graph");
        let mut result = vec![];
        let mut ignore = HashSet::new();

        candidates.extend(self.neighborhood(point).map(|n| {
            let np = self.get(n).unwrap();
            Distance::new(np.distance(p), n, np)
        }));

        while let Some(candidate) = candidates.pop_min() {
            if ignore.contains(&candidate.key) {
                continue;
            }

            result.push(candidate.clone());
            if result.len() >= max_degree {
                break;
            }

            for other in candidates.iter() {
                // if other intersection p is not a subset of candidate
                if other
                    .point
                    .labels
                    .intersection(&p.labels)
                    .any(|i| !candidate.point.labels.contains(i))
                {
                    continue;
                }

                if alpha * candidate.point.distance(other.point) > other.distance {
                    ignore.insert(other.key);
                }
            }
        }

        result
    }
}
