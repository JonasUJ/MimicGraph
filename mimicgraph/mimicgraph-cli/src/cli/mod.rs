pub mod commands;
pub mod utils;

use crate::cli::utils::require_labels;
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
    fn read_labels(
        h5file: &H5File,
        group_name: &str,
        count: usize,
    ) -> anyhow::Result<Vec<LabelSet>> {
        let group = h5file.group(group_name)?;
        let indptr: Vec<usize> = group.dataset("indptr")?.read_1d()?.to_vec();
        let indices: Vec<usize> = group.dataset("indices")?.read_1d()?.to_vec();

        let mut labels = Vec::with_capacity(count);
        for window in indptr.windows(2).take(count) {
            let mut set = LabelSet::new();
            for &idx in &indices[window[0]..window[1]] {
                set.insert(idx);
            }
            labels.push(set);
        }
        Ok(labels)
    }

    pub fn resolve(
        mode: &DatasetMode,
        h5file: &H5File,
        num_corpus: usize,
        num_queries: usize,
    ) -> anyhow::Result<Self> {
        match mode {
            DatasetMode::Unfiltered => Ok(FilteredMode::Unfiltered),
            DatasetMode::Filtered => {
                let labels =
                    require_labels(Self::read_labels(h5file, "labels", num_corpus), "labels")?;
                let query_labels = require_labels(
                    Self::read_labels(h5file, "query_labels", num_queries),
                    "query_labels",
                )?;
                Ok(FilteredMode::Filtered {
                    labels,
                    query_labels,
                })
            }
            DatasetMode::Auto => {
                // Probe whether the datasets exist before doing the expensive read
                let has_labels = h5file.group("labels").is_ok();
                let has_query_labels = h5file.group("query_labels").is_ok();

                if has_labels && has_query_labels {
                    let labels =
                        require_labels(Self::read_labels(h5file, "labels", num_corpus), "labels")?;
                    let query_labels = require_labels(
                        Self::read_labels(h5file, "query_labels", num_queries),
                        "query_labels",
                    )?;
                    Ok(FilteredMode::Filtered {
                        labels,
                        query_labels,
                    })
                } else {
                    tracing::warn!(
                        "Falling back to unfiltered mode because labels/query_labels were not available"
                    );
                    Ok(FilteredMode::Unfiltered)
                }
            }
        }
    }
}
