use crate::artifacts::{WithMetadata, build_and_save, load_or_create};
use crate::cli::utils::*;
use crate::eval::{FilteredTestIndex, TestIndex, compute_ground_truth};
use anyhow::Result;
use hnsw_itu::{HNSWBuilder, IndexBuilder};
use mimicgraph_core::labels::LabelSet;
use mimicgraph_core::mimicgraph::Builder;
use mimicgraph_core::mimicgraph::filtered::{FilteredMimicGraphBuilder, FilteredMimicGraphOptions};
use mimicgraph_core::mimicgraph::plain::MimicGraphBuilder;
use mimicgraph_core::vamana::filtered::{FilteredVamanaBuilder, FilteredVamanaOptions};
use roargraph::Row;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::path::Path;
use std::time::Duration;
use tracing::{error, info};

/// Configuration for which indices to build
pub struct IndexConfig {
    pub build_mimicgraph: bool,
    pub build_hnsw: bool,
    pub build_roargraph: bool,
    pub build_filtered_mimicgraph: bool,
    pub build_filtered_vamana: bool,
}

/// Context for a single build run
pub struct BuildContext<'a> {
    pub artifact_dir: &'a Path,
    pub dataset_name: &'a str,
    pub num_corpus: usize,
    pub corpus: &'a [Row<f32>],
    pub queries: &'a [Row<f32>],
    pub force_recreate: bool,
    pub index_config: IndexConfig,
}

impl<'a> BuildContext<'a> {
    fn artifact<T: Serialize + DeserializeOwned>(
        &self,
        path: &Path,
        create: impl FnOnce() -> T,
    ) -> WithMetadata<T> {
        if self.force_recreate {
            build_and_save(path, create)
        } else {
            load_or_create(path, create)
        }
    }

    fn ensure_enough_queries(&self, build_count: usize, q: f32, index_name: &str) -> Result<()> {
        if self.queries.len() < build_count {
            let msg = format!(
                "Not enough queries for {index_name} build count (q={q}% = {build_count}, queries available: {})",
                self.queries.len()
            );
            error!("{msg}");

            return Err(anyhow::anyhow!(msg));
        }

        Ok(())
    }

    pub fn build_unfiltered(
        &self,
        mg_options_str: &str,
        hnsw_options_str: &str,
        rg_options_str: &str,
    ) -> Result<Vec<(&'static str, String, TestIndex<Row<f32>>, Duration)>> {
        let mut indices = Vec::new();

        if self.index_config.build_mimicgraph {
            let mg_options = parse_mimicgraph_options(mg_options_str)?;

            validate_build_percent(mg_options.q)?;
            let build_count = build_count_from_percent(self.corpus.len(), mg_options.q);
            self.ensure_enough_queries(build_count, mg_options.q, "MimicGraph")?;

            let graph_file = self.artifact_dir.join(format!(
                "mimicgraph_{}_d={}_q={}_{:?}.bin",
                self.dataset_name, self.num_corpus, build_count, mg_options
            ));
            let graph_meta = self.artifact(&graph_file, || {
                info!("Building MimicGraph...");
                MimicGraphBuilder::new(mg_options).build(
                    &self
                        .queries
                        .iter()
                        .take(build_count)
                        .cloned()
                        .collect::<Vec<_>>(),
                    self.corpus.to_vec(),
                )
            });

            indices.push((
                "MimicGraph",
                mg_options_str.to_string(),
                TestIndex::MimicGraph(graph_meta.value),
                graph_meta.build_time,
            ));
        }

        if self.index_config.build_hnsw {
            let (ef_construction, connections, max_connections) =
                parse_hnsw_options(hnsw_options_str)?;
            let hnsw_options = hnsw_itu::NSWOptions {
                ef_construction,
                connections,
                max_connections,
            };

            let hnsw_file = self.artifact_dir.join(format!(
                "hnsw_{}_d={}_{:?}.bin",
                self.dataset_name, self.num_corpus, hnsw_options
            ));
            let hnsw_meta = self.artifact(&hnsw_file, || {
                info!("Building HNSW...");
                let mut builder = HNSWBuilder::new(hnsw_options);
                builder.extend_parallel(self.corpus.iter().cloned());
                builder.build()
            });

            indices.push((
                "HNSW",
                hnsw_options_str.to_string(),
                TestIndex::Hnsw(hnsw_meta.value),
                hnsw_meta.build_time,
            ));
        }

        if self.index_config.build_roargraph {
            let rg_options = parse_roargraph_options(rg_options_str)?;

            validate_build_percent(rg_options.q)?;
            let build_count = build_count_from_percent(self.corpus.len(), rg_options.q);
            self.ensure_enough_queries(build_count, rg_options.q, "RoarGraph")?;

            let build_gt_file = self.artifact_dir.join(format!(
                "build_ground_truth_{}_d={}_q={}.bin",
                self.dataset_name, self.num_corpus, build_count
            ));
            let build_gt = compute_ground_truth(
                path_str(&build_gt_file)?,
                &self.queries[..build_count],
                self.corpus,
            );

            let rg_file = self.artifact_dir.join(format!(
                "roargraph_{}_d={}_q={}_{:?}.bin",
                self.dataset_name, self.num_corpus, build_count, rg_options
            ));
            let rg_meta = self.artifact(&rg_file, || {
                info!("Building RoarGraph (build count: {build_count})...");
                roargraph::RoarGraphBuilder::new(rg_options).build(
                    self.queries.iter().take(build_count).cloned().collect(),
                    self.corpus.to_vec(),
                    build_gt
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
                rg_options_str.to_string(),
                TestIndex::RoarGraph(rg_meta.value),
                rg_meta.build_time + build_gt.build_time,
            ));
        }

        Ok(indices)
    }

    pub fn build_filtered(
        &self,
        labels: &[LabelSet],
        query_labels: &[LabelSet],
        filtered_mg_options_str: &str,
        vamana_options_str: &str,
    ) -> Result<Vec<(&'static str, String, FilteredTestIndex<Row<f32>>, Duration)>> {
        let mut indices = Vec::new();

        if self.index_config.build_filtered_mimicgraph {
            let filtered_mg = parse_filtered_mimicgraph_options(filtered_mg_options_str)?;

            validate_build_percent(filtered_mg.base.q)?;
            let build_count = build_count_from_percent(self.corpus.len(), filtered_mg.base.q);
            self.ensure_enough_queries(build_count, filtered_mg.base.q, "FilteredMimicGraph")?;

            let graph_options = FilteredMimicGraphOptions {
                base_options: filtered_mg.base,
                threshold: filtered_mg.threshold,
                labels: labels.to_vec(),
                query_labels: query_labels[..build_count].to_vec(),
            };

            let graph_file = self.artifact_dir.join(format!(
                "filtered-mimicgraph_{}_d={}_q={}_{:?}.bin",
                self.dataset_name, self.num_corpus, build_count, graph_options.base_options
            ));
            let graph_meta = self.artifact(&graph_file, || {
                info!("Building FilteredMimicGraph...");
                FilteredMimicGraphBuilder::new(graph_options).build(
                    &self
                        .queries
                        .iter()
                        .take(build_count)
                        .cloned()
                        .collect::<Vec<_>>(),
                    self.corpus.to_vec(),
                )
            });

            indices.push((
                "F-MimicGraph",
                filtered_mg_options_str.to_string(),
                FilteredTestIndex::MimicGraph(graph_meta.value),
                graph_meta.build_time,
            ));
        }

        if self.index_config.build_filtered_vamana {
            let vamana_cfg = parse_filtered_vamana_options(vamana_options_str)?;
            let vamana_options = FilteredVamanaOptions {
                alpha: vamana_cfg.alpha,
                l: vamana_cfg.l,
                r: vamana_cfg.r,
                threshold: vamana_cfg.threshold,
                labels: labels.to_vec(),
            };

            let vamana_file = self.artifact_dir.join(format!(
                "filtered-vamana_{}_d={}_{:?}.bin",
                self.dataset_name, self.num_corpus, vamana_options
            ));
            let vamana_meta = self.artifact(&vamana_file, || {
                info!("Building FilteredVamana...");
                let mut builder = FilteredVamanaBuilder::new(vamana_options);
                builder.extend(self.corpus.iter().cloned());
                builder.build()
            });

            indices.push((
                "F-Vamana",
                vamana_options_str.to_string(),
                FilteredTestIndex::Vamana(vamana_meta.value),
                vamana_meta.build_time,
            ));
        }

        Ok(indices)
    }
}
