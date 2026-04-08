use hnsw_itu::{Distance, Point};
use min_max_heap::MinMaxHeap;
use roargraph::AdjListGraph;
use std::collections::{HashMap, HashSet};

pub mod filtered;
pub mod index;
pub mod stitched;

trait GraphExt<P> {
    fn robust_prune<'a>(
        &'a self,
        point: usize,
        candidates: &mut MinMaxHeap<Distance<'a, P>>,
        labels: &HashMap<usize, HashSet<usize>>,
        alpha: f32,
        max_degree: usize,
    ) -> Vec<Distance<'a, P>>;
}

impl<P: Point> GraphExt<P> for AdjListGraph<P> {
    fn robust_prune<'a>(
        &'a self,
        point: usize,
        candidates: &mut MinMaxHeap<Distance<'a, P>>,
        labels: &HashMap<usize, HashSet<usize>>,
        alpha: f32,
        max_degree: usize,
    ) -> Vec<Distance<'a, P>> {
        let p = self.get(point).expect("point not in graph");
        let p_labels = labels.get(&point).expect("labels not found for point");
        let mut result = vec![];
        let mut ignore = HashSet::new();

        candidates.extend(self.neighborhood(point).map(|n| {
            let np = self.get(n).unwrap();
            Distance::new(np.distance(p), n, np)
        }));

        while let Some(candidate) = candidates.pop_min() {
            let candidate_labels = labels
                .get(&candidate.key)
                .expect("labels not found for point");
            if ignore.contains(&candidate.key) {
                continue;
            }

            result.push(candidate.clone());
            if result.len() >= max_degree {
                break;
            }

            for other in candidates.iter() {
                let other_labels = labels.get(&other.key).expect("labels not found for point");
                // if other intersection p is not a subset of candidate
                if other_labels
                    .intersection(p_labels)
                    .any(|i| !candidate_labels.contains(i))
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
