use crate::cli::utils::{dataset_file_name, path_str};
use crate::cli::{DatasetMode, FilteredMode};
use crate::eval::{compute_filtered_ground_truth, compute_ground_truth};
use anyhow::Result;
use clap::Args;
use roargraph::{BufferedDataset, H5File, Row};
use std::fs;
use std::path::PathBuf;
use tracing::info;

/// Compute ground truth for given queries
#[derive(Args, Debug)]
pub struct GroundTruthCommand {
    /// HDF5 file with corpus, queries, and optional label datasets
    #[arg(short, long)]
    datafile: PathBuf,

    /// Directory where artifacts are written
    #[arg(short, long, default_value = "data.exclude")]
    artifact_dir: PathBuf,

    /// How to handle label datasets
    #[arg(long, value_enum, default_value_t = DatasetMode::Auto)]
    dataset_mode: DatasetMode,

    /// Recompute ground truth even if it already exists
    #[arg(long, default_value_t = false)]
    force_recreate: bool,

    /// Max number of corpus points to use
    #[arg(long, default_value_t = 1_000_000)]
    num_corpus: usize,

    /// Number of queries to compute ground truth for
    #[arg(long, default_value_t = 10_000)]
    eval_count: usize,
}

impl GroundTruthCommand {
    pub fn run(self) -> Result<()> {
        fs::create_dir_all(&self.artifact_dir)?;

        let path = self.datafile.as_path();
        let dataset_name = dataset_file_name(path)?;
        info!("Using dataset {path:?}");

        let h5file = H5File::open(path)?;
        let dataset = BufferedDataset::<'_, Row<f32>, _>::open(path, "points")
            .or_else(|_| BufferedDataset::<'_, Row<f32>, _>::open(path, "train"))?;
        let num_corpus = self.num_corpus.min(dataset.size());
        let corpus = dataset.into_iter().take(num_corpus).collect::<Vec<_>>();

        let queries = BufferedDataset::<'_, Row<f32>, _>::open(path, "query_points")
            .or_else(|_| BufferedDataset::<'_, Row<f32>, _>::open(path, "learn"))?
            .into_iter()
            .collect::<Vec<_>>();

        let eval_count = self.eval_count.min(queries.len());
        let eval_start = queries.len().saturating_sub(eval_count);
        let eval_queries = queries[eval_start..].to_vec();

        match FilteredMode::resolve(&self.dataset_mode, &h5file, num_corpus, queries.len())? {
            FilteredMode::Filtered {
                labels,
                query_labels,
            } => {
                let file = self.artifact_dir.join(format!(
                    "filtered_ground_truth_{}_d={}_q={}.bin",
                    dataset_name, num_corpus, eval_count
                ));

                if self.force_recreate && file.exists() {
                    fs::remove_file(&file)?;
                }

                compute_filtered_ground_truth(
                    path_str(&file)?,
                    &eval_queries,
                    &corpus,
                    &labels,
                    &query_labels[eval_start..],
                );
            }
            FilteredMode::Unfiltered => {
                let file = self.artifact_dir.join(format!(
                    "ground_truth_{}_d={}_q={}.bin",
                    dataset_name, num_corpus, eval_count
                ));

                if self.force_recreate && file.exists() {
                    fs::remove_file(&file)?;
                }

                compute_ground_truth(path_str(&file)?, &eval_queries, &corpus);
            }
        }

        Ok(())
    }
}
