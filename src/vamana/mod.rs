use hnsw_itu::{Distance, Point};
use min_max_heap::MinMaxHeap;
use rand::seq::IteratorRandom;
use roargraph::AdjListGraph;
use std::collections::{HashMap, HashSet};
use crate::labels::LabelledPoint;

pub mod filtered;
pub mod index;
pub mod stitched;


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
