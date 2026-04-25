use crate::cli::commands::common::{BuildContext, IndexConfig};
use crate::cli::utils::{dataset_file_name, parse_search_options, path_str};
use crate::cli::{DatasetMode, FilteredMode};
use crate::eval::{
    compute_filtered_ground_truth, compute_ground_truth, evaluate, evaluate_filtered,
};
use anyhow::Result;
use clap::Args;
use roargraph::{BufferedDataset, H5File, Row};
use std::fs;
use std::path::PathBuf;
use tracing::info;

/// Build index artifacts and evaluate recall
#[derive(Args, Debug)]
pub struct EvalCommand {
    /// HDF5 file with corpus, queries, and optional label datasets
    #[arg(short, long)]
    datafile: PathBuf,

    /// Directory where all artifacts are written
    #[arg(short, long, default_value = "data.exclude")]
    artifact_dir: PathBuf,

    /// How to handle label datasets
    #[arg(long, value_enum, default_value_t = DatasetMode::Auto)]
    dataset_mode: DatasetMode,

    /// Rebuild graph artifacts even if they already exist
    #[arg(long, default_value_t = false)]
    force_recreate: bool,

    /// Max number of corpus points to use
    #[arg(long, default_value_t = 1_000_000)]
    num_corpus: usize,

    /// Number of queries used for recall evaluation
    #[arg(long, default_value_t = 10_000)]
    eval_count: usize,

    /// Search comma-separated k:ef pairs
    #[arg(long, default_value = "10:10,10:20,10:100,100:100,100:200,100:1000")]
    search_options: String,

    // Unfiltered index flags (default all true for backward compat, but user can disable)
    /// Build MimicGraph index
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    build_mimicgraph: bool,

    /// Build HNSW index
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    build_hnsw: bool,

    /// Build RoarGraph index
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    build_roargraph: bool,

    /// Build FilteredMimicGraph index
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    build_filtered_mimicgraph: bool,

    /// Build FilteredVamana index
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    build_filtered_vamana: bool,

    /// MimicGraph options (m,l,p,e,qk,qef,con,vis,q)
    #[arg(
        long,
        default_value = "m=32,l=500,p=100,e=16,qk=0,qef=100,con=false,vis=true,q=10"
    )]
    mimicgraph_options: String,

    /// Filtered MimicGraph options (m,l,p,e,qk,qef,con,vis,q,threshold)
    #[arg(
        long,
        default_value = "m=32,l=500,p=100,e=16,qk=0,qef=100,con=false,vis=true,q=10,threshold=1000"
    )]
    filtered_mimicgraph_options: String,

    /// HNSW options (ef_construction,connections,max_connections)
    #[arg(
        long,
        default_value = "ef_construction=400,connections=24,max_connections=64"
    )]
    hnsw_options: String,

    /// Filtered Vamana options (alpha,l,r,threshold)
    #[arg(long, default_value = "alpha=1.2,l=90,r=96,threshold=1000")]
    filtered_vamana_options: String,

    /// RoarGraph options (m,l,q)
    #[arg(long, default_value = "m=32,l=500,q=10")]
    roargraph_options: String,
}

impl EvalCommand {
    pub fn run(self) -> Result<()> {
        fs::create_dir_all(&self.artifact_dir)?;

        let path = self.datafile.as_path();
        let dataset_name = dataset_file_name(path)?;
        info!("Using dataset {path:?}");

        let h5file = H5File::open(path)?;
        let dataset = BufferedDataset::<'_, Row<f32>, _>::open(path, "points")
            .or_else(|_| BufferedDataset::<'_, Row<f32>, _>::open(path, "train"))?;

        let num_corpus = self.num_corpus.min(dataset.size());
        info!("Corpus size: {} of {}", num_corpus, dataset.size());
        let corpus = dataset.into_iter().take(num_corpus).collect::<Vec<_>>();

        let queries = BufferedDataset::<'_, Row<f32>, _>::open(path, "query_points")
            .or_else(|_| BufferedDataset::<'_, Row<f32>, _>::open(path, "learn"))?
            .into_iter()
            .collect::<Vec<_>>();

        let eval_count = self.eval_count.min(queries.len());
        let eval_start = queries.len().saturating_sub(eval_count);
        let eval_queries = queries[eval_start..].to_vec();

        let params = parse_search_options(&self.search_options)?;

        let ctx = BuildContext {
            artifact_dir: &self.artifact_dir,
            dataset_name,
            num_corpus,
            corpus: &corpus,
            queries: &queries,
            force_recreate: self.force_recreate,
            index_config: IndexConfig {
                build_mimicgraph: self.build_mimicgraph,
                build_hnsw: self.build_hnsw,
                build_roargraph: self.build_roargraph,
                build_filtered_mimicgraph: self.build_filtered_mimicgraph,
                build_filtered_vamana: self.build_filtered_vamana,
            },
        };

        match FilteredMode::resolve(&self.dataset_mode, &h5file, num_corpus, queries.len())? {
            FilteredMode::Filtered {
                labels,
                query_labels,
            } => {
                let eval_query_labels = &query_labels[eval_start..];

                let gt_file = self.artifact_dir.join(format!(
                    "filtered_ground_truth_{}_d={}_q={}.bin",
                    dataset_name, num_corpus, eval_count
                ));
                let ground_truth = compute_filtered_ground_truth(
                    path_str(&gt_file)?,
                    &eval_queries,
                    &corpus,
                    &labels,
                    eval_query_labels,
                );

                let indices = ctx.build_filtered(
                    &labels,
                    &query_labels,
                    &self.filtered_mimicgraph_options,
                    &self.filtered_vamana_options,
                )?;

                evaluate_filtered(
                    indices,
                    &params,
                    &eval_queries,
                    &ground_truth.value,
                    eval_query_labels,
                );
            }
            FilteredMode::Unfiltered => {
                let gt_file = self.artifact_dir.join(format!(
                    "ground_truth_{}_d={}_q={}.bin",
                    dataset_name, num_corpus, eval_count
                ));
                let ground_truth =
                    compute_ground_truth(path_str(&gt_file)?, &queries[eval_start..], &corpus);

                let indices = ctx.build_unfiltered(
                    &self.mimicgraph_options,
                    &self.hnsw_options,
                    &self.roargraph_options,
                )?;

                evaluate(indices, &params, &eval_queries, &ground_truth.value);
            }
        }

        Ok(())
    }
}
