use crate::eval::{
    FilteredTestIndex, TestIndex, compute_filtered_ground_truth, compute_ground_truth, evaluate,
    evaluate_filtered,
};
use crate::labels::LabelSet;
use crate::mimicgraph::filtered::{FilteredMimicGraphBuilder, FilteredMimicGraphOptions};
use crate::mimicgraph::plain::MimicGraphBuilder;
use crate::mimicgraph::{Builder, MimicGraphOptions};
use crate::vamana::filtered::{FilteredVamanaBuilder, FilteredVamanaOptions};
use bincode::{deserialize_from, serialize_into};
use hnsw_itu::{HNSWBuilder, IndexBuilder, NSWOptions};
use roargraph::{BufferedDataset, H5File, RoarGraphOptions, Row};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sprs::CsMat;
use std::fs::File;
use std::io::Write;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::{Duration, Instant};
use tracing::info;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

pub mod bitset;
pub mod eval;
pub mod labels;
pub mod mimicgraph;
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
    let path = Path::new("../datasets/data.exclude/LAION1M.hdf5");
    //let path = Path::new("../datasets/data.exclude/yfcc.hdf5");
    //let path = Path::new("../datasets/data.exclude/tripclick.hdf5");
    //let path = Path::new("../datasets/data.exclude/arxiv.hdf5");
    //let path = Path::new("../datasets/data.exclude/YFCC-10M.hdf5");
    let outdir = "data.exclude";

    info!("Using dataset {path:?}");
    let h5file = H5File::open(path)?;
    let dataset = BufferedDataset::<'_, Row<f32>, _>::open(path, "points")
        .or_else(|_| BufferedDataset::<'_, Row<f32>, _>::open(path, "train"))?;
    //let num_corpus = 500_000.min(dataset.size());
    let num_corpus = 1_000_000.min(dataset.size());
    //let num_corpus = 10_000_000.min(dataset.size());
    info!("Corpus size: {} of {}", num_corpus, dataset.size());
    let corpus = dataset.into_iter().take(num_corpus).collect::<Vec<_>>();
    let build_count = corpus.len() / 10;
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

    let options = MimicGraphOptions {
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

        let options = FilteredMimicGraphOptions {
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
            FilteredMimicGraphBuilder::new(options).build(
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
            "mimicgraph-{}.txt",
            path.file_name().unwrap().to_str().unwrap()
        );
        let mut file = File::create(outfile)?;
        for p in graph.inner.graph.adj_lists().iter() {
            writeln!(file, "{}", p.len())?;
        }

        indices.push((
            "F-MimicGraph",
            FilteredTestIndex::MimicGraph(graph),
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
            MimicGraphBuilder::new(options).build(
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
            "mimicgraph-{}.txt",
            path.file_name().unwrap().to_str().unwrap()
        );
        let mut file = File::create(outfile)?;
        for p in graph.graph.adj_lists().iter() {
            writeln!(file, "{}", p.len())?;
        }

        indices.push((
            "MimicGraph",
            TestIndex::MimicGraph(graph),
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

#[derive(Serialize, Deserialize)]
pub struct WithMetadata<T> {
    pub value: T,
    pub build_time: Duration,
}

impl<T> WithMetadata<T> {
    pub fn new(value: T, build_time: Duration) -> Self {
        Self { value, build_time }
    }
}
