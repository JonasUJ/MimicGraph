use crate::artifacts::{WithMetadata, inspect_with_metadata};
use anyhow::Result;
use bincode::deserialize_from;
use clap::Args;
use hnsw_itu::HNSW;
use mimicgraph_core::mimicgraph::filtered::FilteredMimicGraph;
use mimicgraph_core::mimicgraph::plain::MimicGraph;
use mimicgraph_core::vamana::index::FilteredVamana;
use roargraph::{AdjListGraph, RoarGraph, Row};
use serde::de::DeserializeOwned;
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
    if let Ok(metadata) = try_read::<MimicGraph<Row<f32>>>(path) {
        return write_degree_file_from_graph(path, artifact_dir, &metadata.value.graph).map(Some);
    }

    if let Ok(metadata) = try_read::<FilteredMimicGraph<Row<f32>>>(path) {
        return write_degree_file_from_graph(path, artifact_dir, &metadata.value.inner.graph)
            .map(Some);
    }

    if let Ok(metadata) = try_read::<HNSW<Row<f32>>>(path) {
        let out = degree_file_path(path, artifact_dir)?;
        let mut file = File::create(&out)?;

        for adj in metadata.value.graph().adj_lists().iter() {
            writeln!(file, "{}", adj.len())?;
        }

        return Ok(Some(out));
    }

    if let Ok(metadata) = try_read::<RoarGraph<Row<f32>>>(path) {
        return write_degree_file_from_graph(path, artifact_dir, &metadata.value.graph).map(Some);
    }

    if let Ok(metadata) = try_read::<FilteredVamana<Row<f32>>>(path) {
        return write_degree_file_from_graph(path, artifact_dir, &metadata.value.graph).map(Some);
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

fn write_degree_file_from_graph<T>(
    path: &Path,
    artifact_dir: &Path,
    graph: &AdjListGraph<T>,
) -> Result<PathBuf> {
    let out = degree_file_path(path, artifact_dir)?;
    let mut file = File::create(&out)?;

    for adj in graph.adj_lists().iter() {
        writeln!(file, "{}", adj.len())?;
    }

    Ok(out)
}
