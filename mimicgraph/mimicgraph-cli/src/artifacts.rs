use anyhow::Result;
use bincode::{deserialize_from, serialize_into};
use hnsw_itu::HNSW;
use mimicgraph_core::mimicgraph::filtered::FilteredMimicGraph;
use mimicgraph_core::mimicgraph::plain::MimicGraph;
use mimicgraph_core::vamana::index::FilteredVamana;
use roargraph::{RoarGraph, Row};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::{Duration, Instant};
use tracing::info;

/// Load artifact from disk if it exists, otherwise build and save it.
pub fn load_or_create<T: Serialize + DeserializeOwned>(
    path: &Path,
    create: impl FnOnce() -> T,
) -> WithMetadata<T> {
    if path.exists() {
        info!("Reading {path:?}");
        let reader = BufReader::new(File::open(path).unwrap());
        deserialize_from(reader).unwrap()
    } else {
        build_and_save(path, create)
    }
}

pub fn build_and_save<T: Serialize + DeserializeOwned>(
    path: &Path,
    create: impl FnOnce() -> T,
) -> WithMetadata<T> {
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

#[derive(Debug)]
pub struct InspectedMetadata {
    #[allow(dead_code)]
    pub value_type: &'static str,
    #[allow(dead_code)]
    pub build_time: Duration,
}

pub fn inspect_with_metadata(path: &Path) -> Result<InspectedMetadata> {
    macro_rules! probe {
        ($ty:ty, $name:expr) => {
            if let Ok(metadata) = try_read_metadata::<$ty>(path) {
                return Ok(InspectedMetadata {
                    value_type: $name,
                    build_time: metadata.build_time,
                });
            }
        };
    }

    probe!(Vec<Vec<(usize, f32)>>, "Vec<Vec<(usize,f32)>>");
    probe!(MimicGraph<Row<f32>>, "MimicGraph<Row<f32>>");
    probe!(FilteredMimicGraph<Row<f32>>, "FilteredMimicGraph<Row<f32>>");
    probe!(HNSW<Row<f32>>, "HNSW<Row<f32>>");
    probe!(RoarGraph<Row<f32>>, "RoarGraph<Row<f32>>");
    probe!(FilteredVamana<Row<f32>>, "FilteredVamana<Row<f32>>");

    Err(anyhow::anyhow!(
        "could not deserialize `{}` as a known artifact type",
        path.display()
    ))
}

fn try_read_metadata<T: DeserializeOwned>(path: &Path) -> Result<WithMetadata<T>> {
    let reader = BufReader::new(File::open(path)?);
    Ok(deserialize_from(reader)?)
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
