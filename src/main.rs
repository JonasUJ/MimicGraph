#![allow(unused)]

use crate::thesisindex::{Searchable, ThesisIndex, ThesisIndexBuilder, ThesisIndexOptions};
use bincode::{deserialize_from, serialize_into};
use hnsw_itu::{Distance, HNSW, HNSWBuilder, Index, IndexBuilder, MinK, NSWOptions, Point};
use ndarray::Array1;
use rayon::prelude::*;
use roargraph::BufferedDataset;
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
    let path = Path::new("../datasets/data.exclude/laion-clip-512-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/coco-nomic-768-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/llama-128-ip.hdf5");
    //let path = Path::new("../datasets/data.exclude/yi-128-ip.hdf5");
    //let path = Path::new("../datasets/data.exclude/imagenet-align-640-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/imagenet-clip-512-normalized.hdf5");
    let outdir = "data.exclude";

    info!("Using dataset {path:?}");
    let dataset = BufferedDataset::<'_, Row<f32>, _>::open(path, "train").unwrap();
    //let num_corpus = 250_000.min(dataset.size());
    let num_corpus = 10_000_000.min(dataset.size());
    info!("Corpus size: {} of {}", num_corpus, dataset.size());
    let corpus = dataset.into_iter().take(num_corpus).collect::<Vec<_>>();
    let build_count = corpus.len() / 50;
    let eval_count = 10_000;
    let queries = BufferedDataset::<'_, Row<f32>, _>::open(path, "learn")
        .unwrap()
        .into_iter()
        .collect::<Vec<_>>();

    let options = ThesisIndexOptions {
        m: 32,
        l: 500,
        p: 300,
        e: 16,
        qk: 0,
        qef: 400,
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
    let graph = create_if_not_exists(graph_file, || {
        info!("Building index...");
        let start = Instant::now();
        let index = ThesisIndexBuilder::new(options).build(
            &queries.iter().take(build_count).cloned().collect(),
            corpus.iter().cloned().collect(),
        );
        let elapsed = start.elapsed();
        info!("Index build time: {:?}", elapsed);

        index
    });

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

    let ground_truth = compute_ground_truth(&ground_truth_file_name, &queries, &corpus, eval_count);

    let eval_queries = queries
        .iter()
        .skip(queries.len() - eval_count)
        .cloned()
        .collect::<Vec<_>>();

    let mut indices = vec![("ThesisIndex", TestIndex::Thesis(graph))];

    if true {
        let options = NSWOptions {
            ef_construction: 300,
            connections: 12,
            max_connections: 32,
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
            let start = Instant::now();
            let mut builder = HNSWBuilder::new(options);
            builder.extend_parallel(corpus.iter().cloned());
            let hnsw = builder.build();
            let elapsed = start.elapsed();
            info!("HNSW build time: {:?}", elapsed);

            hnsw
        });
        indices.push(("HNSW", TestIndex::HNSW(hnsw)));
    }

    info!("Evaluating recall (eval queries: {})...", eval_queries.len());
    let params = vec![(10, 10), (10, 20), (10, 100), (100, 100), (100, 200), (100, 1000)];
    let mut recalls = Vec::new();
    let mut spqs = Vec::new();
    for (name, index) in indices {
        let (recall, spq) = evaluate_recall(&eval_queries, &ground_truth, &params, &index);
        recalls.push((name, recall));
        spqs.push((name, spq));
    }

    let header: String = params
        .iter()
        .map(|(k, ef)| format!("{:>15}", format!("k={k},ef={ef}")))
        .collect::<Vec<_>>()
        .join("");
    println!(" {:>12}  {}", "", header);
    println!("Recall");
    for (name, recall) in recalls {
        let row: String = recall
            .iter()
            .map(|r| format!("{:>15.4}", r))
            .collect::<Vec<_>>()
            .join("");
        println!(" {:>12}: {}", name, row);
    }
    println!("SPQ");
    for (name, spq) in spqs {
        let row: String = spq
            .iter()
            .map(|s| format!("{:>15}", format!("{s:?}")))
            .collect::<Vec<_>>()
            .join("");
        println!(" {:>12}: {}", name, row);
    }
}

fn compute_ground_truth(
    filename: &str,
    queries: &Vec<Row<f32>>,
    corpus: &Vec<Row<f32>>,
    eval_count: usize,
) -> Vec<Vec<(usize, f32)>> {
    let ground_truth_file = Path::new(filename);
    let ground_truth = create_if_not_exists(ground_truth_file, || {
        info!("Computing ground truth nearest neighbors...");
        queries
            .par_iter()
            .skip(queries.len() - eval_count)
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
            .collect::<Vec<_>>()
    });

    ground_truth
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
) -> T {
    if path.exists() {
        info!("Reading {path:?}");
        let reader = BufReader::new(File::open(&path).unwrap());
        deserialize_from(reader).unwrap()
    } else {
        info!("Creating {path:?}");
        let result = create();
        let writer = BufWriter::new(File::create(path).unwrap());

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
}

impl<P: Point> Index<P> for TestIndex<P> {
    type Options = usize;
    type Query = P;

    fn size(&self) -> usize {
        match self {
            TestIndex::Thesis(index) => index.size(),
            TestIndex::HNSW(index) => index.size(),
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
        }
    }
}
