use anyhow::Result;
use bincode::{deserialize_from, serialize_into};
use hnsw_itu::{HNSW, Idx, SimpleGraph};
use mimicgraph_core::bitset::Bitset;
use mimicgraph_core::labels::LabelSet;
use mimicgraph_core::mimicgraph::filtered::FilteredMimicGraph;
use mimicgraph_core::mimicgraph::plain::MimicGraph;
use mimicgraph_core::vamana::index::FilteredVamana;
use roargraph::{AdjListGraph, RoarGraph, Row};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::info;

pub trait Topology: Sized {
    type Compact: Serialize + DeserializeOwned;

    fn into_topology(self) -> Self::Compact;

    fn from_topology(compact: Self::Compact, data: Vec<Row<f32>>) -> Self;
}

#[derive(Serialize, Deserialize)]
pub struct MimicGraphTopology {
    pub entry: usize,
    pub adj_lists: Vec<HashSet<usize>>,
}

impl Topology for MimicGraph<Row<f32>> {
    type Compact = MimicGraphTopology;

    fn into_topology(self) -> MimicGraphTopology {
        let (_, adj_lists) = self.graph.consume();
        MimicGraphTopology {
            entry: self.entry,
            adj_lists,
        }
    }

    fn from_topology(compact: MimicGraphTopology, data: Vec<Row<f32>>) -> Self {
        MimicGraph {
            entry: compact.entry,
            graph: AdjListGraph {
                nodes: data,
                adj_lists: compact.adj_lists,
            },
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct RoarGraphTopology {
    pub medoid: usize,
    pub adj_lists: Vec<HashSet<usize>>,
}

impl Topology for RoarGraph<Row<f32>> {
    type Compact = RoarGraphTopology;

    fn into_topology(self) -> RoarGraphTopology {
        let medoid = self.medoid();
        let (_, adj_lists) = self.graph.consume();
        RoarGraphTopology { medoid, adj_lists }
    }

    fn from_topology(compact: RoarGraphTopology, data: Vec<Row<f32>>) -> Self {
        RoarGraph::from_parts(
            compact.medoid,
            AdjListGraph {
                nodes: data,
                adj_lists: compact.adj_lists,
            },
        )
    }
}

#[derive(Serialize, Deserialize)]
pub struct HNSWTopology {
    pub layer_adj_lists: Vec<(Vec<Idx>, Vec<HashSet<Idx>>)>,
    pub base_adj_lists: Vec<HashSet<Idx>>,
    pub ep: Option<Idx>,
}

impl Topology for HNSW<Row<f32>> {
    type Compact = HNSWTopology;

    fn into_topology(self) -> HNSWTopology {
        let (layers, base, ep) = self.consume();

        let layer_adj_lists = layers
            .into_iter()
            .map(|sg| {
                let (nodes, adj_lists) = sg.consume();
                let indices: Vec<Idx> = nodes.into_iter().map(|(_, idx)| idx).collect();
                (indices, adj_lists)
            })
            .collect();

        let (_, base_adj_lists) = base.consume();

        HNSWTopology {
            layer_adj_lists,
            base_adj_lists,
            ep,
        }
    }

    fn from_topology(compact: HNSWTopology, data: Vec<Row<f32>>) -> Self {
        let layers: Vec<SimpleGraph<(Row<f32>, Idx)>> = compact
            .layer_adj_lists
            .into_iter()
            .map(|(indices, adj_lists)| {
                let nodes: Vec<(Row<f32>, Idx)> = indices
                    .into_iter()
                    .map(|idx| (data[idx].clone(), idx))
                    .collect();
                SimpleGraph::from_parts(nodes, adj_lists)
            })
            .collect();

        let base = SimpleGraph::from_parts(data, compact.base_adj_lists);

        HNSW::from_parts(layers, base, compact.ep)
    }
}

#[derive(Serialize, Deserialize)]
pub struct FilteredVamanaTopology {
    pub start_nodes: HashMap<usize, usize>,
    pub adj_lists: Vec<HashSet<usize>>,
    pub labels: Vec<LabelSet>,
}

impl Topology for FilteredVamana<Row<f32>> {
    type Compact = FilteredVamanaTopology;

    fn into_topology(self) -> FilteredVamanaTopology {
        let (_, adj_lists) = self.graph.consume();
        FilteredVamanaTopology {
            start_nodes: self.start_nodes,
            adj_lists,
            labels: self.labels,
        }
    }

    fn from_topology(compact: FilteredVamanaTopology, data: Vec<Row<f32>>) -> Self {
        FilteredVamana {
            start_nodes: compact.start_nodes,
            graph: AdjListGraph {
                nodes: data,
                adj_lists: compact.adj_lists,
            },
            labels: compact.labels,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct FilteredMimicGraphTopology {
    pub inner: FilteredVamanaTopology,
    pub inverted_index: Vec<Bitset>,
}

impl Topology for FilteredMimicGraph<Row<f32>> {
    type Compact = FilteredMimicGraphTopology;

    fn into_topology(self) -> FilteredMimicGraphTopology {
        FilteredMimicGraphTopology {
            inner: self.inner.into_topology(),
            inverted_index: self.inverted_index,
        }
    }

    fn from_topology(compact: FilteredMimicGraphTopology, data: Vec<Row<f32>>) -> Self {
        FilteredMimicGraph {
            inner: FilteredVamana::from_topology(compact.inner, data),
            inverted_index: compact.inverted_index,
        }
    }
}

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
    let result = WithMetadata::new(result, elapsed, None);
    serialize_into(writer, &result).unwrap();

    result
}

pub fn build_and_save_topology<I: Topology>(
    path: &Path,
    dataset_path: &Path,
    create: impl FnOnce() -> I,
) -> WithMetadata<I::Compact> {
    info!("Creating {path:?}");
    let start = Instant::now();
    let index = create();
    let elapsed = start.elapsed();
    info!("Build time: {:?}", elapsed);

    let compact = index.into_topology();

    let writer = BufWriter::new(File::create(path).unwrap());
    let result = WithMetadata::new(compact, elapsed, Some(dataset_path.to_path_buf()));
    serialize_into(writer, &result).unwrap();

    result
}

pub fn load_or_create_topology<I: Topology>(
    path: &Path,
    dataset_path: &Path,
    create: impl FnOnce() -> I,
) -> WithMetadata<I::Compact> {
    if path.exists() {
        info!("Reading {path:?}");
        let reader = BufReader::new(File::open(path).unwrap());
        deserialize_from(reader).unwrap()
    } else {
        build_and_save_topology(path, dataset_path, create)
    }
}

#[derive(Debug)]
pub struct InspectedMetadata {
    #[allow(dead_code)]
    pub value_type: &'static str,
    #[allow(dead_code)]
    pub build_time: Duration,
    #[allow(dead_code)]
    pub dataset_path: Option<PathBuf>,
}

pub fn inspect_with_metadata(path: &Path) -> Result<InspectedMetadata> {
    macro_rules! probe {
        ($ty:ty, $name:expr) => {
            if let Ok(metadata) = try_read_metadata::<$ty>(path) {
                return Ok(InspectedMetadata {
                    value_type: $name,
                    build_time: metadata.build_time,
                    dataset_path: metadata.dataset_path,
                });
            }
        };
    }

    probe!(Vec<Vec<(usize, f32)>>, "Vec<Vec<(usize,f32)>>");
    probe!(MimicGraphTopology, "MimicGraph");
    probe!(RoarGraphTopology, "RoarGraph");
    probe!(HNSWTopology, "HNSW");
    probe!(FilteredMimicGraphTopology, "FilteredMimicGraph");
    probe!(FilteredVamanaTopology, "FilteredVamana");

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
    pub dataset_path: Option<PathBuf>,
}

impl<T> WithMetadata<T> {
    pub fn new(value: T, build_time: Duration, dataset_path: Option<PathBuf>) -> Self {
        Self {
            value,
            build_time,
            dataset_path,
        }
    }
}
