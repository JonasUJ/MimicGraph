pub mod commands;
pub mod utils;

use crate::cli::utils::{csmat_to_map, require_labels};
use clap::{Parser, Subcommand, ValueEnum};
use mimicgraph_core::labels::LabelSet;
use roargraph::H5File;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(flatten)]
    pub verbose: clap_verbosity_flag::Verbosity<clap_verbosity_flag::InfoLevel>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Eval(commands::eval::EvalCommand),
    Build(commands::build::BuildCommand),
    GroundTruth(commands::ground_truth::GroundTruthCommand),
    Inspect(commands::inspect::InspectCommand),
}

impl Commands {
    pub fn exec(self) -> anyhow::Result<()> {
        match self {
            Self::Eval(cmd) => cmd.run(),
            Self::Build(cmd) => cmd.run(),
            Self::GroundTruth(cmd) => cmd.run(),
            Self::Inspect(cmd) => cmd.run(),
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum DatasetMode {
    Auto,
    Filtered,
    Unfiltered,
}

#[derive(Copy, Clone, Debug, Default, ValueEnum)]
pub enum OutputFormat {
    #[default]
    Table,
    Csv,
}

/// Resolved filtered/unfiltered state with loaded label data.
pub enum FilteredMode {
    Filtered {
        labels: Vec<LabelSet>,
        query_labels: Vec<LabelSet>,
    },
    Unfiltered,
}

impl FilteredMode {
    pub fn resolve(
        mode: &DatasetMode,
        h5file: &H5File,
        num_corpus: usize,
        num_queries: usize,
    ) -> anyhow::Result<Self> {
        let labels_result = h5file
            .read_csr::<usize>("labels")
            .map(|m| csmat_to_map(m, num_corpus));
        let query_labels_result = h5file
            .read_csr::<usize>("query_labels")
            .map(|m| csmat_to_map(m, num_queries));

        let use_filtered = match mode {
            DatasetMode::Filtered => true,
            DatasetMode::Unfiltered => false,
            DatasetMode::Auto => labels_result.is_ok() && query_labels_result.is_ok(),
        };

        if matches!(mode, DatasetMode::Auto) && !use_filtered {
            tracing::warn!(
                "Falling back to unfiltered mode because labels/query_labels were not available"
            );
        }

        if use_filtered {
            Ok(FilteredMode::Filtered {
                labels: require_labels(labels_result, "labels")?,
                query_labels: require_labels(query_labels_result, "query_labels")?,
            })
        } else {
            Ok(FilteredMode::Unfiltered)
        }
    }
}
