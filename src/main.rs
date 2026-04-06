#![allow(unused)]

use crate::thesisindex::{Searchable, ThesisIndex, ThesisIndexBuilder, ThesisIndexOptions};
use bincode::{deserialize_from, serialize_into};
use hnsw_itu::{Distance, HNSW, HNSWBuilder, Index, IndexBuilder, MinK, NSWOptions, Point};
use ndarray::Array1;
use rayon::prelude::*;
use roargraph::{BufferedDataset, RoarGraph, RoarGraphOptions};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::{Duration, Instant};
use tracing::info;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

mod thesisindex;
mod vamana;

fn main() {
    tracing_subscriber::registry().with(fmt::layer()).init();

    //let path = Path::new("../datasets/data.exclude/arxiv-nomic-768-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/laion-clip-512-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/coco-nomic-768-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/llama-128-ip.hdf5");
    //let path = Path::new("../datasets/data.exclude/yi-128-ip.hdf5");
    let path = Path::new("../datasets/data.exclude/imagenet-align-640-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/imagenet-clip-512-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/LAION1M.hdf5");
    //let path = Path::new("../datasets/data.exclude/YFCC-10M.hdf5");
    let outdir = "data.exclude";

    info!("Using dataset {path:?}");
    let dataset = BufferedDataset::<'_, Row<f32>, _>::open(path, "points")
        .or_else(|_| BufferedDataset::<'_, Row<f32>, _>::open(path, "train"))
        .unwrap();
    let num_corpus = 250_000.min(dataset.size());
    //let num_corpus = 1_000_000.min(dataset.size());
    //let num_corpus = 10_000_000.min(dataset.size());
    info!("Corpus size: {} of {}", num_corpus, dataset.size());
    let corpus = dataset.into_iter().take(num_corpus).collect::<Vec<_>>();
    let build_count = corpus.len() / 20;
    let eval_count = 10_000;
    let queries = BufferedDataset::<'_, Row<f32>, _>::open(path, "query_points")
        .or_else(|_| BufferedDataset::<'_, Row<f32>, _>::open(path, "learn"))
        .unwrap()
        .into_iter()
        .collect::<Vec<_>>();

    let options = ThesisIndexOptions {
        m: 32,
        l: 500,
        p: 100,
        e: 8,
        qk: 0,
        qef: 100,
        con: false,
        vis: true,
    };
    info!("{options:?}");
    info!(
        "Query count: {build_count} ({:.3}% of corpus)",
        build_count as f64 * 100.0 / num_corpus as f64
    );

    let graph_file_name = format!(
        "{outdir}/graph_{}_d={num_corpus}_q={build_count}_{:?}.bin",
        path.file_name().unwrap().to_str().unwrap(),
        options
    );
    let graph_file = Path::new(graph_file_name.as_str());
    let graph_metadata = create_if_not_exists(graph_file, || {
        info!("Building index...");
        let index = ThesisIndexBuilder::new(options).build(
            &queries.iter().take(build_count).cloned().collect(),
            corpus.iter().cloned().collect(),
        );

        index
    });

    let graph = graph_metadata.value;

    let outfile = format!(
        "thesisindex-{}.txt",
        path.file_name().unwrap().to_str().unwrap()
    );
    let mut file = File::create(outfile).unwrap();
    for p in graph.graph.adj_lists().into_iter() {
        writeln!(file, "{}", p.len()).unwrap();
    }

    let ground_truth_file_name = format!(
        "{outdir}/ground_truth_{}_d={}_q={}.bin",
        path.file_name().unwrap().to_str().unwrap(),
        num_corpus,
        eval_count,
    );

    let ground_truth = compute_ground_truth(
        &ground_truth_file_name,
        &queries[queries.len() - eval_count..],
        &corpus,
    );

    let mut indices = vec![(
        "ThesisIndex",
        TestIndex::Thesis(graph),
        graph_metadata.build_time,
    )];

    let eval_queries = queries
        .iter()
        .skip(queries.len() - eval_count)
        .cloned()
        .collect::<Vec<_>>();

    if true {
        let options = NSWOptions {
            ef_construction: 400,
            connections: 24,
            max_connections: 64,
            size: corpus.len(),
        };
        let hnsw_file_name = format!(
            "{outdir}/hnsw_{}_d={num_corpus}_{:?}.bin",
            path.file_name().unwrap().to_str().unwrap(),
            options
        );
        let hnsw_file = Path::new(hnsw_file_name.as_str());
        let hnsw = create_if_not_exists(hnsw_file, || {
            info!("Building HNSW index...");
            let mut builder = HNSWBuilder::new(options);
            builder.extend_parallel(corpus.iter().cloned());
            let hnsw = builder.build();

            hnsw
        });
        indices.push(("HNSW", TestIndex::HNSW(hnsw.value), hnsw.build_time));
    }

    if true {
        let options = RoarGraphOptions { m: 32, l: 500 };

        let build_gt_file_name = format!(
            "{outdir}/build_ground_truth_{}_d={}_q={}.bin",
            path.file_name().unwrap().to_str().unwrap(),
            num_corpus,
            build_count,
        );
        let build_ground_truth =
            compute_ground_truth(&build_gt_file_name, &queries[..build_count], &corpus);

        let roargraph_file_name = format!(
            "{outdir}/roargraph_{}_d={num_corpus}_q={build_count}_{:?}.bin",
            path.file_name().unwrap().to_str().unwrap(),
            options
        );
        let roargraph_file = Path::new(roargraph_file_name.as_str());
        let roargraph = create_if_not_exists(roargraph_file, || {
            info!("Building RoarGraph index (query count: {build_count})...");
            let roargraph = roargraph::RoarGraphBuilder::new(options).build(
                queries.iter().take(build_count).cloned().collect(),
                corpus.iter().cloned().collect(),
                build_ground_truth
                    .value
                    .iter()
                    .take(build_count)
                    .cloned()
                    .map(|v| v.into_iter().map(|(i, _)| i).collect())
                    .collect(),
            );

            roargraph
        });
        indices.push((
            "RoarGraph",
            TestIndex::RoarGraph(roargraph.value),
            roargraph.build_time + build_ground_truth.build_time,
        ));
    }

    info!(
        "Evaluating recall (eval queries: {})...",
        eval_queries.len()
    );
    let params = vec![
        (10, 10),
        (10, 20),
        (10, 100),
        (100, 100),
        (100, 200),
        (100, 1000),
    ];
    let mut recalls = Vec::new();
    let mut spqs = Vec::new();
    let mut build_times = Vec::new();
    for (name, index, build_time) in indices {
        let (recall, spq) = evaluate_recall(&eval_queries, &ground_truth.value, &params, &index);
        recalls.push((name, recall));
        spqs.push((name, spq));
        build_times.push((name, build_time));
    }

    let header: String = params
        .iter()
        .map(|(k, ef)| format!("{:>15}", format!("k={k},ef={ef}")))
        .collect::<Vec<_>>()
        .join("");
    println!(" {:>12}  {:>15} {}", "", "Build Time (BT)", header);
    print_rows("Recall", &recalls, &build_times, |r| format!("{:>15.4}", r));
    print_rows("SPQ", &spqs, &build_times, |s| {
        format!("{:>15}", fmt_duration(s))
    });

    let qps: Vec<(&str, Vec<f64>)> = spqs
        .iter()
        .map(|(name, vals)| (*name, vals.iter().map(|s| 1.0 / s.as_secs_f64()).collect()))
        .collect();
    print_rows("QPS", &qps, &build_times, |r| format!("{:>15.1}", r));

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
    print_rows("Recall/BT", &recall_per_bt, &build_times, |r| {
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
    print_rows("QPS/BT", &qps_per_bt, &build_times, |r| {
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

fn compute_ground_truth(
    filename: &str,
    queries: &[Row<f32>],
    corpus: &Vec<Row<f32>>,
) -> WithMetadata<Vec<Vec<(usize, f32)>>> {
    create_if_not_exists(Path::new(filename), || {
        info!("Computing ground truth nearest neighbors...");
        let ground_truth = queries
            .par_iter()
            .map(|q| {
                let mut closest = corpus
                    .iter()
                    .enumerate()
                    .map(|(k, d)| Distance::new(d.distance(&q), k, d))
                    .min_k(100);
                closest.sort();
                closest
                    .iter()
                    .map(|d| (d.key, d.distance))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        ground_truth
    })
}

fn evaluate_recall(
    queries: &Vec<Row<f32>>,
    ground_truth: &Vec<Vec<(usize, f32)>>,
    params: &Vec<(usize, usize)>,
    index: &TestIndex<Row<f32>>,
) -> (Vec<f32>, Vec<Duration>) {
    let ground_truth_keys = ground_truth
        .iter()
        .map(|v| v.iter().map(|(k, _)| *k).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let eval_ground_truth = ground_truth_keys
        .iter()
        .cloned()
        .zip(queries.iter())
        .collect::<Vec<_>>();

    let mut recalls = Vec::new();
    let mut spqs = Vec::new();
    for (k, ef) in params {
        let result: Vec<(f32, Duration)> = eval_ground_truth
            .par_iter()
            .map(|(knn, query)| {
                let knn = knn.iter().copied().collect::<HashSet<_>>();

                let start = Instant::now();
                let mut found = index.search(query, *k, &ef);
                let elapsed = start.elapsed();
                let found = found
                    .drain(..)
                    .take(*k)
                    .map(|d| d.key)
                    .collect::<HashSet<_>>();

                (knn.intersection(&found).count() as f32 / *k as f32, elapsed)
            })
            .collect();

        let recall = result.iter().map(|(r, _)| r).sum::<f32>() / eval_ground_truth.len() as f32;
        let spq = result.iter().map(|(_, e)| e).sum::<Duration>() / eval_ground_truth.len() as u32;
        recalls.push(recall);
        spqs.push(spq);
    }

    (recalls, spqs)
}

fn create_if_not_exists<T: Serialize + DeserializeOwned, C: FnOnce() -> T>(
    path: &Path,
    create: C,
) -> WithMetadata<T> {
    if path.exists() {
        info!("Reading {path:?}");
        let reader = BufReader::new(File::open(&path).unwrap());
        deserialize_from(reader).unwrap()
    } else {
        info!("Creating {path:?}");
        let start = Instant::now();
        let result = create();
        let elapsed = start.elapsed();
        info!("Build time: {:?}", elapsed);
        let writer = BufWriter::new(File::create(path).unwrap());

        let result = WithMetadata::new(result, elapsed);
        serialize_into(writer, &result).unwrap();
        result
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Row<T> {
    data: Vec<T>,
}

impl<T: Clone> From<Array1<T>> for Row<T> {
    fn from(value: Array1<T>) -> Self {
        Self {
            data: value.to_vec(),
        }
    }
}

impl Point for Row<f32> {
    fn distance(&self, other: &Self) -> f32 {
        // Inner product distance
        -self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f32>()
    }
}

enum TestIndex<P> {
    Thesis(ThesisIndex<P>),
    HNSW(HNSW<P>),
    RoarGraph(RoarGraph<P>),
}

impl<P: Point> Index<P> for TestIndex<P> {
    type Options = usize;
    type Query = P;

    fn size(&self) -> usize {
        match self {
            TestIndex::Thesis(index) => index.size(),
            TestIndex::HNSW(index) => index.size(),
            TestIndex::RoarGraph(index) => index.size(),
        }
    }

    fn search(
        &'_ self,
        query: &Self::Query,
        k: usize,
        options: &Self::Options,
    ) -> Vec<Distance<'_, P>>
    where
        P: Point,
    {
        match self {
            TestIndex::Thesis(index) => index.search(query, k, options),
            TestIndex::HNSW(index) => index.search(query, k, options),
            TestIndex::RoarGraph(index) => {
                index.search(query, *options).drain_asc().take(k).collect()
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct WithMetadata<T> {
    value: T,
    build_time: Duration,
}

impl<T> WithMetadata<T> {
    pub fn new(value: T, build_time: Duration) -> Self {
        Self { value, build_time }
    }
}
