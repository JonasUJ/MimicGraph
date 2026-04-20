use crate::WithMetadata;
use hnsw_itu::{Distance, HNSW, Index, MinK, Point};
use mimicgraph_core::labels::LabelSet;
use mimicgraph_core::mimicgraph::filtered::{FilteredMimicGraph, FilteredMimicGraphSearchOptions};
use mimicgraph_core::mimicgraph::plain::MimicGraph;
use mimicgraph_core::vamana::index::{FilteredVamana, FilteredVamanaSearchOptions};
use rayon::prelude::*;
use roargraph::{RoarGraph, Row};
use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};
use tracing::info;

pub enum TestIndex<P> {
    MimicGraph(MimicGraph<P>),
    Hnsw(HNSW<P>),
    RoarGraph(RoarGraph<P>),
}

impl<P: Point> Index<P> for TestIndex<P> {
    type Options<'a> = usize;

    fn size(&self) -> usize {
        match self {
            TestIndex::MimicGraph(index) => index.size(),
            TestIndex::Hnsw(index) => index.size(),
            TestIndex::RoarGraph(index) => index.size(),
        }
    }

    fn search(&'_ self, query: &P, k: usize, options: &Self::Options<'_>) -> Vec<Distance<'_, P>>
    where
        P: Point,
    {
        match self {
            TestIndex::MimicGraph(index) => index.search(query, k, options),
            TestIndex::Hnsw(index) => index.search(query, k, options),
            TestIndex::RoarGraph(index) => index.search(query, k, options),
        }
    }
}

pub enum FilteredTestIndex<P> {
    MimicGraph(FilteredMimicGraph<P>),
    Vamana(FilteredVamana<P>),
}

impl<P: Point> Index<P> for FilteredTestIndex<P> {
    type Options<'a> = FilteredMimicGraphSearchOptions<'a>;

    fn size(&self) -> usize {
        match self {
            FilteredTestIndex::MimicGraph(index) => index.size(),
            FilteredTestIndex::Vamana(index) => index.size(),
        }
    }

    fn search(&'_ self, query: &P, k: usize, options: &Self::Options<'_>) -> Vec<Distance<'_, P>>
    where
        P: Point,
    {
        match self {
            FilteredTestIndex::MimicGraph(index) => index.search(query, k, options),
            FilteredTestIndex::Vamana(index) => {
                let vamana_options = FilteredVamanaSearchOptions {
                    ef: options.ef,
                    labels: options.labels,
                };
                index.search(query, k, &vamana_options)
            }
        }
    }
}

pub fn evaluate(
    indices: Vec<(&str, TestIndex<Row<f32>>, Duration)>,
    params: &[(usize, usize)],
    eval_queries: &[Row<f32>],
    ground_truth: &[Vec<(usize, f32)>],
) {
    info!(
        "Evaluating recall (eval queries: {}, threads: {})...",
        eval_queries.len(),
        rayon::current_num_threads()
    );

    let mut recalls = Vec::new();
    let mut spqs = Vec::new();
    let mut build_times = Vec::new();
    for (name, index, build_time) in indices {
        let mut index_recalls = Vec::new();
        let mut index_spqs = Vec::new();
        for &(k, ef) in params {
            let (recall, spq) = evaluate_recall(eval_queries, ground_truth, k, &|_qi, query, k| {
                index
                    .search(query, k, &ef)
                    .into_iter()
                    .map(|d| d.key)
                    .collect()
            });
            index_recalls.push(recall);
            index_spqs.push(spq);
        }
        recalls.push((name, index_recalls));
        spqs.push((name, index_spqs));
        build_times.push((name, build_time));
    }

    let header: Vec<String> = params
        .iter()
        .map(|(k, ef)| format!("k={k},ef={ef}"))
        .collect();
    print_evaluation_results(&header, &recalls, &spqs, &build_times);
}

pub fn evaluate_filtered(
    indices: Vec<(&str, FilteredTestIndex<Row<f32>>, Duration)>,
    params: &[(usize, usize)],
    eval_queries: &[Row<f32>],
    ground_truth: &[Vec<(usize, f32)>],
    query_labels: &[LabelSet],
) {
    info!(
        "Evaluating filtered recall (eval queries: {}, threads: {})...",
        eval_queries.len(),
        rayon::current_num_threads()
    );

    let mut recalls = Vec::new();
    let mut spqs = Vec::new();
    let mut build_times = Vec::new();
    for (name, index, build_time) in indices {
        let mut index_recalls = Vec::new();
        let mut index_spqs = Vec::new();
        for &(k, ef) in params {
            let (recall, spq) = evaluate_recall(eval_queries, ground_truth, k, &|qi, query, k| {
                let options = FilteredMimicGraphSearchOptions {
                    ef,
                    labels: &query_labels[qi],
                    scan_limit: 0,
                };
                index
                    .search(query, k, &options)
                    .into_iter()
                    .map(|d| d.key)
                    .collect()
            });
            index_recalls.push(recall);
            index_spqs.push(spq);
        }
        recalls.push((name, index_recalls));
        spqs.push((name, index_spqs));
        build_times.push((name, build_time));
    }

    let header: Vec<String> = params
        .iter()
        .map(|(k, ef)| format!("k={k},ef={ef}"))
        .collect();
    print_evaluation_results(&header, &recalls, &spqs, &build_times);
}

fn print_evaluation_results(
    header_labels: &[String],
    recalls: &[(&str, Vec<f32>)],
    spqs: &[(&str, Vec<Duration>)],
    build_times: &[(&str, Duration)],
) {
    let header: String = header_labels
        .iter()
        .map(|l| format!("{:>15}", l))
        .collect::<Vec<_>>()
        .join("");
    println!(" {:>12}  {:>15} {}", "", "Build Time (BT)", header);
    print_rows("Recall", recalls, build_times, |r| format!("{:>15.4}", r));
    print_rows("SPQ", spqs, build_times, |s| {
        format!("{:>15}", fmt_duration(s))
    });

    let qps: Vec<(&str, Vec<f64>)> = spqs
        .iter()
        .map(|(name, vals)| (*name, vals.iter().map(|s| 1.0 / s.as_secs_f64()).collect()))
        .collect();
    print_rows("QPS", &qps, build_times, |r| format!("{:>15.1}", r));

    let recall_per_bt: Vec<(&str, Vec<f64>)> = recalls
        .iter()
        .map(|(name, vals)| {
            let bt = build_times
                .iter()
                .find(|(n, _)| n == name)
                .unwrap()
                .1
                .as_secs_f64();
            (*name, vals.iter().map(|r| *r as f64 / bt).collect())
        })
        .collect();
    print_rows("Recall/BT", &recall_per_bt, build_times, |r| {
        format!("{:>15.6}", r)
    });

    let qps_per_bt: Vec<(&str, Vec<f64>)> = spqs
        .iter()
        .map(|(name, vals)| {
            let bt = build_times
                .iter()
                .find(|(n, _)| n == name)
                .unwrap()
                .1
                .as_secs_f64();
            (
                *name,
                vals.iter().map(|s| 1.0 / (s.as_secs_f64() * bt)).collect(),
            )
        })
        .collect();
    print_rows("QPS/BT", &qps_per_bt, build_times, |r| {
        format!("{:>15.1}", r)
    });
}

fn fmt_duration(d: &Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 1.0 {
        format!("{:.2}s", secs)
    } else if secs >= 1e-3 {
        format!("{:.2}ms", secs * 1e3)
    } else {
        format!("{:.2}µs", secs * 1e6)
    }
}

fn print_rows<T>(
    label: &str,
    rows: &[(&str, Vec<T>)],
    build_times: &[(&str, Duration)],
    fmt: impl Fn(&T) -> String,
) {
    println!("{label}");
    for (name, values) in rows {
        let bt = build_times
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| fmt_duration(t))
            .unwrap_or_default();
        let row: String = values.iter().map(&fmt).collect::<Vec<_>>().join("");
        println!(" {:>12}: {:>15} {}", name, bt, row);
    }
}

pub fn compute_ground_truth(
    filename: &str,
    queries: &[Row<f32>],
    corpus: &[Row<f32>],
) -> WithMetadata<Vec<Vec<(usize, f32)>>> {
    crate::create_if_not_exists(Path::new(filename), || {
        info!("Computing ground truth nearest neighbors...");
        queries
            .par_iter()
            .map(|q| {
                let mut closest = corpus
                    .iter()
                    .enumerate()
                    .map(|(k, d)| Distance::new(d.distance(q), k, d))
                    .min_k(100);
                closest.sort();
                closest
                    .iter()
                    .map(|d| (d.key, d.distance))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    })
}

pub fn compute_filtered_ground_truth(
    filename: &str,
    queries: &[Row<f32>],
    corpus: &[Row<f32>],
    labels: &[LabelSet],
    query_labels: &[LabelSet],
) -> WithMetadata<Vec<Vec<(usize, f32)>>> {
    crate::create_if_not_exists(Path::new(filename), || {
        info!("Computing filtered ground truth nearest neighbors...");
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
                        if mimicgraph_core::labels::labels_intersect(d_labels, q_labels) {
                            Some(Distance::new(d.distance(q), di, d))
                        } else {
                            None
                        }
                    })
                    .min_k(100);
                closest.sort();
                closest
                    .iter()
                    .map(|d| (d.key, d.distance))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    })
}

fn evaluate_recall(
    queries: &[Row<f32>],
    ground_truth: &[Vec<(usize, f32)>],
    k: usize,
    search: &(impl Fn(usize, &Row<f32>, usize) -> Vec<usize> + Sync),
) -> (f32, Duration) {
    let ground_truth_keys: Vec<Vec<usize>> = ground_truth
        .iter()
        .map(|v| v.iter().map(|(k, _)| *k).collect())
        .collect();

    let result: Vec<(f32, Duration)> = ground_truth_keys
        .par_iter()
        .enumerate()
        .map(|(qi, knn)| {
            let knn: HashSet<usize> = knn.iter().copied().collect();
            let query = &queries[qi];

            let start = Instant::now();
            let found_keys = search(qi, query, k);
            let elapsed = start.elapsed();
            let found: HashSet<usize> = found_keys.into_iter().take(k).collect();

            (knn.intersection(&found).count() as f32 / k as f32, elapsed)
        })
        .collect();

    let recall = result.iter().map(|(r, _)| r).sum::<f32>() / result.len() as f32;
    let spq = result.iter().map(|(_, e)| e).sum::<Duration>() / result.len() as u32;
    (recall, spq)
}
