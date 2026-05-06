use crate::artifacts::{Topology, WithMetadata, build_and_save_index, load_index};
use crate::cli::utils::*;
use crate::eval::{FilteredTestIndex, TestIndex, compute_ground_truth};
use anyhow::Result;
use hnsw_itu::{HNSWBuilder, IndexBuilder};
use mimicgraph_core::labels::LabelSet;
use mimicgraph_core::mimicgraph::filtered::FilteredMimicGraphBuilder;
use mimicgraph_core::mimicgraph::plain::MimicGraphBuilder;
use mimicgraph_core::mimicgraph::{Builder, FilteredMimicGraphOptions, MimicGraphOptions};
use mimicgraph_core::vamana::filtered::{FilteredVamanaBuilder, FilteredVamanaOptions};
use roargraph::Row;
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
    pub dataset_path: &'a Path,
    pub num_corpus: usize,
    pub corpus: Vec<Row<f32>>,
    pub queries: &'a [Row<f32>],
    pub force_recreate: bool,
    pub index_config: IndexConfig,
}

impl BuildContext<'_> {
    /// Load or build an index artifact.
    /// When loading from disk the corpus is moved (no clone).
    /// When building, the corpus is cloned for the builder; the original is
    /// moved into from_topology for reconstruction.
    fn load_or_build<I: Topology>(
        &self,
        path: &Path,
        corpus: Vec<Row<f32>>,
        create: impl FnOnce(Vec<Row<f32>>) -> I,
    ) -> WithMetadata<I> {
        if !self.force_recreate && path.exists() {
            load_index(path, corpus)
        } else {
            let build_corpus = corpus.clone();
            build_and_save_index(path, self.dataset_path, corpus, || create(build_corpus))
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

    /// Take the corpus if this is the last index to build, otherwise clone.
    fn take_or_clone_corpus(&mut self, is_last: bool) -> Vec<Row<f32>> {
        if is_last {
            std::mem::take(&mut self.corpus)
        } else {
            self.corpus.clone()
        }
    }

    pub fn build_unfiltered(
        mut self,
        mg_options_str: &str,
        hnsw_options_str: &str,
        rg_options_str: &str,
    ) -> Result<Vec<(&'static str, String, TestIndex<Row<f32>>, Duration)>> {
        let mut indices = Vec::new();

        // Count how many indices will be built to know when to move vs clone
        let remaining = [
            self.index_config.build_mimicgraph,
            self.index_config.build_hnsw,
            self.index_config.build_roargraph,
        ];
        let total = remaining.iter().filter(|&&b| b).count();
        let mut built = 0;

        if self.index_config.build_mimicgraph {
            built += 1;
            let is_tuned = mg_options_str.trim().eq_ignore_ascii_case("tuned");
            let mg_options = if is_tuned {
                MimicGraphOptions::tuned(&self.corpus, self.queries)
            } else {
                parse_mimicgraph_options(mg_options_str)?
            };

            validate_build_percent(mg_options.q)?;
            let build_count = build_count_from_percent(self.corpus.len(), mg_options.q);
            self.ensure_enough_queries(build_count, mg_options.q, "MimicGraph")?;

            let options_label = if is_tuned {
                "tuned".to_string()
            } else {
                format!("{:?}", mg_options)
            };
            let graph_file = self.artifact_dir.join(format!(
                "mimicgraph_{}_d={}_q={}_{}.bin",
                self.dataset_name, self.num_corpus, build_count, options_label
            ));
            let queries: Vec<_> = self.queries.iter().take(build_count).cloned().collect();
            let corpus = self.take_or_clone_corpus(built == total);
            let meta = self.load_or_build(&graph_file, corpus, |c| {
                info!("Building MimicGraph...");
                MimicGraphBuilder::new(mg_options).build(&queries, c)
            });

            indices.push((
                "MimicGraph",
                mg_options_str.to_string(),
                TestIndex::MimicGraph(meta.value),
                meta.build_time,
            ));
        }

        if self.index_config.build_hnsw {
            built += 1;
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
            let corpus = self.take_or_clone_corpus(built == total);
            let meta = self.load_or_build(&hnsw_file, corpus, |c| {
                info!("Building HNSW...");
                let mut builder = HNSWBuilder::new(hnsw_options);
                builder.extend_parallel(c);
                builder.build()
            });

            indices.push((
                "HNSW",
                hnsw_options_str.to_string(),
                TestIndex::Hnsw(meta.value),
                meta.build_time,
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
                &self.corpus,
            );

            let rg_file = self.artifact_dir.join(format!(
                "roargraph_{}_d={}_q={}_{:?}.bin",
                self.dataset_name, self.num_corpus, build_count, rg_options
            ));
            let queries: Vec<_> = self.queries.iter().take(build_count).cloned().collect();
            let corpus = std::mem::take(&mut self.corpus); // always last
            let meta = self.load_or_build(&rg_file, corpus, |c| {
                info!("Building RoarGraph (build count: {build_count})...");
                roargraph::RoarGraphBuilder::new(rg_options).build(
                    queries,
                    c,
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
                TestIndex::RoarGraph(meta.value),
                meta.build_time + build_gt.build_time,
            ));
        }

        Ok(indices)
    }

    pub fn build_filtered(
        mut self,
        labels: &[LabelSet],
        query_labels: &[LabelSet],
        filtered_mg_options_str: &str,
        vamana_options_str: &str,
    ) -> Result<Vec<(&'static str, String, FilteredTestIndex<Row<f32>>, Duration)>> {
        let mut indices = Vec::new();

        let total = [
            self.index_config.build_filtered_mimicgraph,
            self.index_config.build_filtered_vamana,
        ]
        .iter()
        .filter(|&&b| b)
        .count();
        let mut built = 0;

        if self.index_config.build_filtered_mimicgraph {
            built += 1;
            let is_tuned = filtered_mg_options_str.trim().eq_ignore_ascii_case("tuned");
            let graph_options = if is_tuned {
                FilteredMimicGraphOptions::tuned(
                    &self.corpus,
                    self.queries,
                    labels.to_vec(),
                    query_labels.to_vec(),
                )
            } else {
                let filtered_mg = parse_filtered_mimicgraph_options(filtered_mg_options_str)?;
                FilteredMimicGraphOptions {
                    base_options: filtered_mg.base,
                    threshold: filtered_mg.threshold,
                    labels: labels.to_vec(),
                    query_labels: query_labels.to_vec(),
                }
            };

            validate_build_percent(graph_options.base_options.q)?;
            let build_count =
                build_count_from_percent(self.corpus.len(), graph_options.base_options.q);
            self.ensure_enough_queries(
                build_count,
                graph_options.base_options.q,
                "FilteredMimicGraph",
            )?;

            let graph_options = FilteredMimicGraphOptions {
                query_labels: query_labels[..build_count].to_vec(),
                ..graph_options
            };

            let options_label = if is_tuned {
                "tuned".to_string()
            } else {
                format!("{:?}", graph_options.base_options)
            };
            let graph_file = self.artifact_dir.join(format!(
                "filtered-mimicgraph_{}_d={}_q={}_{}.bin",
                self.dataset_name, self.num_corpus, build_count, options_label
            ));
            let queries: Vec<_> = self.queries.iter().take(build_count).cloned().collect();
            let corpus = self.take_or_clone_corpus(built == total);
            let meta = self.load_or_build(&graph_file, corpus, |c| {
                info!("Building FilteredMimicGraph...");
                FilteredMimicGraphBuilder::new(graph_options).build(&queries, c)
            });

            indices.push((
                "F-MimicGraph",
                filtered_mg_options_str.to_string(),
                FilteredTestIndex::MimicGraph(meta.value),
                meta.build_time,
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
            let corpus = std::mem::take(&mut self.corpus); // always last
            let meta = self.load_or_build(&vamana_file, corpus, |c| {
                info!("Building FilteredVamana...");
                let mut builder = FilteredVamanaBuilder::new(vamana_options);
                builder.extend(c.into_iter());
                builder.build()
            });

            indices.push((
                "F-Vamana",
                vamana_options_str.to_string(),
                FilteredTestIndex::Vamana(meta.value),
                meta.build_time,
            ));
        }

        Ok(indices)
    }
}
