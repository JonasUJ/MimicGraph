use crate::labels::LabelSet;
use crate::thesis_index::filtered::{
    FilteredThesisIndex, FilteredThesisIndexBuilder, FilteredThesisIndexOptions,
};
use crate::thesis_index::plain::{ThesisIndex, ThesisIndexBuilder};
use crate::thesis_index::{Builder, ThesisIndexOptions};
use crate::vamana::filtered::{FilteredVamanaBuilder, FilteredVamanaOptions};
use crate::vamana::index::{FilteredVamana, FilteredVamanaSearchOptions};
use bincode::{deserialize_from, serialize_into};
use hnsw_itu::{Distance, HNSW, HNSWBuilder, Index, IndexBuilder, MinK, NSWOptions, Point};
use ndarray::Array1;
use rayon::prelude::*;
use roargraph::{BufferedDataset, H5File, RoarGraph, RoarGraphOptions};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sprs::CsMat;
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

pub mod labels;
pub mod thesis_index;
pub mod vamana;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry().with(fmt::layer()).init();

    //let path = Path::new("../datasets/data.exclude/arxiv-nomic-768-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/laion-clip-512-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/coco-nomic-768-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/llama-128-ip.hdf5");
    //let path = Path::new("../datasets/data.exclude/yi-128-ip.hdf5");
    //let path = Path::new("../datasets/data.exclude/imagenet-align-640-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/imagenet-clip-512-normalized.hdf5");
    //let path = Path::new("../datasets/data.exclude/LAION1M.hdf5");
    let path = Path::new("../datasets/data.exclude/arxiv.hdf5");
    //let path = Path::new("../datasets/data.exclude/YFCC-10M.hdf5");
    let outdir = "data.exclude";

    info!("Using dataset {path:?}");
    let h5file = H5File::open(path)?;
    let dataset = BufferedDataset::<'_, Row<f32>, _>::open(path, "points")
        .or_else(|_| BufferedDataset::<'_, Row<f32>, _>::open(path, "train"))?;
    let num_corpus = 250_000.min(dataset.size());
    //let num_corpus = 1_000_000.min(dataset.size());
    //let num_corpus = 10_000_000.min(dataset.size());
    info!("Corpus size: {} of {}", num_corpus, dataset.size());
    let corpus = dataset.into_iter().take(num_corpus).collect::<Vec<_>>();
    let build_count = corpus.len() / 20;
    let eval_count = 10_000;
    let queries = BufferedDataset::<'_, Row<f32>, _>::open(path, "query_points")
        .or_else(|_| BufferedDataset::<'_, Row<f32>, _>::open(path, "learn"))?
        .into_iter()
        .collect::<Vec<_>>();

    fn csmat_to_map(mat: CsMat<usize>, count: usize) -> Vec<LabelSet> {
        let mut map = Vec::with_capacity(mat.rows());
        let indices = mat.indices();
        for window in mat.proper_indptr().windows(2).take(count) {
            let mut set = LabelSet::new();
            for &idx in &indices[window[0]..window[1]] {
                set.insert(idx);
            }
            map.push(set);
        }
        map
    }

    let labels = h5file
        .read_csr::<usize>("labels")
        .map(|m| csmat_to_map(m, num_corpus));
    let query_labels = h5file
        .read_csr::<usize>("query_labels")
        .map(|m| csmat_to_map(m, queries.len()));
    let is_filtered = labels.is_ok();

    let options = ThesisIndexOptions {
        m: 32,
        l: 500,
        p: 100,
        e: 16,
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

    let params = vec![
        (10, 10),
        (10, 20),
        (10, 100),
        (100, 100),
        (100, 200),
        (100, 1000),
    ];

    if is_filtered {
        let mut indices = vec![];

        let labels = labels?;
        let query_labels = query_labels?;

        let eval_queries = queries
            .iter()
            .skip(queries.len() - eval_count)
            .cloned()
            .collect::<Vec<_>>();

        let eval_query_labels = &query_labels[queries.len() - eval_count..];

        let ground_truth_file_name = format!(
            "{outdir}/filtered_ground_truth_{}_d={}_q={}.bin",
            path.file_name().unwrap().to_str().unwrap(),
            num_corpus,
            eval_count,
        );
        let ground_truth = compute_filtered_ground_truth(
            &ground_truth_file_name,
            &eval_queries,
            &corpus,
            &labels,
            eval_query_labels,
        );

        let options = FilteredThesisIndexOptions {
            base_options: options,
            threshold: 1000,
            labels: labels.clone(),
            query_labels: query_labels[..build_count].to_vec(),
        };

        let graph_file_name = format!(
            "{outdir}/filtered-graph_{}_d={num_corpus}_q={build_count}_{:?}.bin",
            path.file_name().unwrap().to_str().unwrap(),
            options.base_options
        );
        let graph_file = Path::new(graph_file_name.as_str());
        let graph_metadata = create_if_not_exists(graph_file, || {
            info!("Building index...");
            FilteredThesisIndexBuilder::new(options).build(
                &queries
                    .iter()
                    .take(build_count)
                    .cloned()
                    .collect::<Vec<_>>(),
                corpus.to_vec(),
            )
        });

        let graph = graph_metadata.value;

        let outfile = format!(
            "thesisindex-{}.txt",
            path.file_name().unwrap().to_str().unwrap()
        );
        let mut file = File::create(outfile)?;
        for p in graph.graph.adj_lists().iter() {
            writeln!(file, "{}", p.len())?;
        }

        indices.push((
            "F-Thesis",
            FilteredTestIndex::Thesis(graph),
            graph_metadata.build_time,
        ));

        if true {
            let vamana_options = FilteredVamanaOptions {
                alpha: 1.2,
                l: 90,
                r: 96,
                threshold: 1000,
                labels: labels.clone(),
            };
            let vamana_file_name = format!(
                "{outdir}/filtered-vamana_{}_d={num_corpus}_{:?}.bin",
                path.file_name().unwrap().to_str().unwrap(),
                vamana_options
            );
            let vamana_file = Path::new(vamana_file_name.as_str());
            let vamana = create_if_not_exists(vamana_file, || {
                info!("Building FilteredVamana index...");
                let mut builder = FilteredVamanaBuilder::new(vamana_options);
                builder.extend(corpus.iter().cloned());
                builder.build()
            });

            indices.push((
                "F-Vamana",
                FilteredTestIndex::Vamana(vamana.value),
                vamana.build_time,
            ));
        }

        evaluate_filtered(
            indices,
            &params,
            &eval_queries,
            &ground_truth.value,
            eval_query_labels,
        );
    } else {
        // non-filtered
        let mut indices = vec![];

        let eval_queries = queries
            .iter()
            .skip(queries.len() - eval_count)
            .cloned()
            .collect::<Vec<_>>();

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

        let graph_file_name = format!(
            "{outdir}/graph_{}_d={num_corpus}_q={build_count}_{:?}.bin",
            path.file_name().unwrap().to_str().unwrap(),
            options
        );
        let graph_file = Path::new(graph_file_name.as_str());
        let graph_metadata = create_if_not_exists(graph_file, || {
            info!("Building index...");
            ThesisIndexBuilder::new(options).build(
                &queries
                    .iter()
                    .take(build_count)
                    .cloned()
                    .collect::<Vec<_>>(),
                corpus.to_vec(),
            )
        });

        let graph = graph_metadata.value;

        let outfile = format!(
            "thesisindex-{}.txt",
            path.file_name().unwrap().to_str().unwrap()
        );
        let mut file = File::create(outfile)?;
        for p in graph.graph.adj_lists().iter() {
            writeln!(file, "{}", p.len())?;
        }

        indices.push((
            "ThesisIndex",
            TestIndex::Thesis(graph),
            graph_metadata.build_time,
        ));

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

                builder.build()
            });
            indices.push(("HNSW", TestIndex::Hnsw(hnsw.value), hnsw.build_time));
        }

        if build_count <= 20_000 {
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
                roargraph::RoarGraphBuilder::new(options).build(
                    queries.iter().take(build_count).cloned().collect(),
                    corpus.to_vec(),
                    build_ground_truth
                        .value
                        .iter()
                        .take(build_count)
                        .cloned()
                        .map(|v| v.into_iter().map(|(i, _)| i).collect())
                        .collect(),
                )
            });
            indices.push((
                "RoarGraph",
                TestIndex::RoarGraph(roargraph.value),
                roargraph.build_time + build_ground_truth.build_time,
            ));
        }

        evaluate(indices, &params, &eval_queries, &ground_truth.value);
    }

    Ok(())
}

fn evaluate(
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

fn evaluate_filtered(
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
                let options = FilteredVamanaSearchOptions {
                    ef,
                    labels: &query_labels[qi],
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

fn compute_ground_truth(
    filename: &str,
    queries: &[Row<f32>],
    corpus: &[Row<f32>],
) -> WithMetadata<Vec<Vec<(usize, f32)>>> {
    create_if_not_exists(Path::new(filename), || {
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

fn compute_filtered_ground_truth(
    filename: &str,
    queries: &[Row<f32>],
    corpus: &[Row<f32>],
    labels: &[LabelSet],
    query_labels: &[LabelSet],
) -> WithMetadata<Vec<Vec<(usize, f32)>>> {
    create_if_not_exists(Path::new(filename), || {
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
                        if crate::labels::labels_intersect(d_labels, q_labels) {
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

fn create_if_not_exists<T: Serialize + DeserializeOwned, C: FnOnce() -> T>(
    path: &Path,
    create: C,
) -> WithMetadata<T> {
    if path.exists() {
        info!("Reading {path:?}");
        let reader = BufReader::new(File::open(path).unwrap());
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
    Hnsw(HNSW<P>),
    RoarGraph(RoarGraph<P>),
}

impl<P: Point> Index<P> for TestIndex<P> {
    type Options<'a> = usize;

    fn size(&self) -> usize {
        match self {
            TestIndex::Thesis(index) => index.size(),
            TestIndex::Hnsw(index) => index.size(),
            TestIndex::RoarGraph(index) => index.size(),
        }
    }

    fn search(&'_ self, query: &P, k: usize, options: &Self::Options<'_>) -> Vec<Distance<'_, P>>
    where
        P: Point,
    {
        match self {
            TestIndex::Thesis(index) => index.search(query, k, options),
            TestIndex::Hnsw(index) => index.search(query, k, options),
            TestIndex::RoarGraph(index) => index.search(query, k, options),
        }
    }
}

enum FilteredTestIndex<P> {
    Thesis(FilteredThesisIndex<P>),
    Vamana(FilteredVamana<P>),
}

impl<P: Point> Index<P> for FilteredTestIndex<P> {
    type Options<'a> = FilteredVamanaSearchOptions<'a>;

    fn size(&self) -> usize {
        match self {
            FilteredTestIndex::Thesis(index) => index.size(),
            FilteredTestIndex::Vamana(index) => index.size(),
        }
    }

    fn search(&'_ self, query: &P, k: usize, options: &Self::Options<'_>) -> Vec<Distance<'_, P>>
    where
        P: Point,
    {
        match self {
            FilteredTestIndex::Thesis(index) => index.search(query, k, options),
            FilteredTestIndex::Vamana(index) => index.search(query, k, options),
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
