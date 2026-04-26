use crate::artifacts::{
    FilteredMimicGraphTopology, FilteredVamanaTopology, HNSWTopology, MimicGraphTopology,
    RoarGraphTopology, WithMetadata, inspect_with_metadata,
};
use anyhow::Result;
use bincode::deserialize_from;
use clap::Args;
use serde::de::DeserializeOwned;
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

/// Inspect artifact files written by mimicgraph-cli
#[derive(Args, Debug)]
pub struct InspectCommand {
    /// Artifact file paths (.bin) written by mimicgraph-cli
    #[arg(value_name = "FILES", required = true)]
    files: Vec<PathBuf>,

    /// Directory where generated artifacts (e.g. degree files) are written
    #[arg(short, long, default_value = "data.exclude")]
    artifact_dir: PathBuf,
}

impl InspectCommand {
    pub fn run(self) -> Result<()> {
        fs::create_dir_all(&self.artifact_dir)?;

        for file in &self.files {
            let metadata = inspect_with_metadata(file)?;

            println!();
            println!("{}", file.display());
            println!("{:?}", metadata);

            if let Some(out) = maybe_write_degree_file(file, &self.artifact_dir)? {
                println!("Wrote degree file: {}", out.display());
            }
        }

        Ok(())
    }
}

fn maybe_write_degree_file(path: &Path, artifact_dir: &Path) -> Result<Option<PathBuf>> {
    if let Ok(metadata) = try_read::<MimicGraphTopology>(path) {
        return write_degree_file(path, artifact_dir, &metadata.value.adj_lists).map(Some);
    }

    if let Ok(metadata) = try_read::<FilteredMimicGraphTopology>(path) {
        return write_degree_file(path, artifact_dir, &metadata.value.inner.adj_lists).map(Some);
    }

    if let Ok(metadata) = try_read::<HNSWTopology>(path) {
        return write_degree_file(path, artifact_dir, &metadata.value.base_adj_lists).map(Some);
    }

    if let Ok(metadata) = try_read::<RoarGraphTopology>(path) {
        return write_degree_file(path, artifact_dir, &metadata.value.adj_lists).map(Some);
    }

    if let Ok(metadata) = try_read::<FilteredVamanaTopology>(path) {
        return write_degree_file(path, artifact_dir, &metadata.value.adj_lists).map(Some);
    }

    Ok(None)
}

fn try_read<T: DeserializeOwned>(path: &Path) -> Result<WithMetadata<T>> {
    let reader = BufReader::new(File::open(path)?);

    Ok(deserialize_from(reader)?)
}

fn degree_file_path(path: &Path, artifact_dir: &Path) -> Result<PathBuf> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("artifact path must include a valid UTF-8 file name"))?;

    Ok(artifact_dir.join(format!("degree_{name}.txt")))
}

fn write_degree_file(
    path: &Path,
    artifact_dir: &Path,
    adj_lists: &[HashSet<usize>],
) -> Result<PathBuf> {
    let out = degree_file_path(path, artifact_dir)?;
    let mut file = File::create(&out)?;

    for adj in adj_lists {
        writeln!(file, "{}", adj.len())?;
    }

    Ok(out)
}
