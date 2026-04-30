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

pub fn fraction_and_spread(
    ground_truth: &[Vec<(usize, f32)>],
    data_size: usize,
    k: usize,
) -> (f32, f32) {
    let mut counts = vec![0usize; data_size];

    for knn in ground_truth.iter() {
        for (i, _) in knn {
            counts[*i] += 1;
        }
    }

    let nonzero_count = counts.iter().filter(|&&i| i != 0).count();

    // Fraction of data points that appear in the ground truth at least once
    let data_fraction = nonzero_count as f32 / data_size as f32;

    // Normalized inverse Simpson index number
    let data_spread = {
        let total = counts.iter().sum::<usize>() as f32;
        let sum_sq: f32 = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f32 / total;
                p * p
            })
            .sum();

        let effective_support = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };
        let min_support = k.min(data_size) as f32;
        let max_support = data_size as f32;

        ((effective_support - min_support) / (max_support - min_support)).clamp(0.0, 1.0)
    };

    (data_fraction, data_spread)
}
