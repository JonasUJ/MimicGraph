use crate::cli::commands::common::{BuildContext, IndexConfig};
use crate::cli::utils::dataset_file_name;
use crate::cli::{DatasetMode, FilteredMode};
use clap::Args;
use roargraph::{BufferedDataset, H5File, Row};
use std::fs;
use std::path::PathBuf;
use tracing::info;

/// Build index artifacts
#[derive(Args, Debug)]
pub struct BuildCommand {
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

impl BuildCommand {
    pub fn run(self) -> anyhow::Result<()> {
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
                ctx.build_filtered(
                    &labels,
                    &query_labels,
                    &self.filtered_mimicgraph_options,
                    &self.filtered_vamana_options,
                )?;
            }
            FilteredMode::Unfiltered => {
                ctx.build_unfiltered(
                    &self.mimicgraph_options,
                    &self.hnsw_options,
                    &self.roargraph_options,
                )?;
            }
        }

        Ok(())
    }
}
