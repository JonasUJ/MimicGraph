use crate::labels;
use crate::labels::LabelSet;
use hnsw_itu::{Distance, MinK, Point};
use rayon::prelude::*;

pub fn compute_ground_truth<P: Point + Sync>(
    queries: &[P],
    corpus: &[P],
    k: usize,
) -> Vec<Vec<(usize, f32)>> {
    queries
        .par_iter()
        .map(|q| {
            let mut closest = corpus
                .iter()
                .enumerate()
                .map(|(i, d)| Distance::new(d.distance(q), i, d))
                .min_k(k);

            closest.sort();

            closest
                .iter()
                .map(|d| (d.key, d.distance))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

pub fn compute_filtered_ground_truth<P: Point + Sync>(
    queries: &[P],
    corpus: &[P],
    labels: &[LabelSet],
    query_labels: &[LabelSet],
    k: usize,
) -> Vec<Vec<(usize, f32)>> {
    queries
        .par_iter()
        .enumerate()
        .map(|(qi, q)| {
            let q_labels = &query_labels[qi];

            let mut closest = corpus
                .iter()
                .enumerate()
                .filter_map(|(di, d)| {
                    let d_labels = &labels[di];

                    if labels::labels_intersect(d_labels, q_labels) {
                        Some(Distance::new(d.distance(q), di, d))
                    } else {
                        None
                    }
                })
                .min_k(k);

            closest.sort();

            closest
                .iter()
                .map(|d| (d.key, d.distance))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}
