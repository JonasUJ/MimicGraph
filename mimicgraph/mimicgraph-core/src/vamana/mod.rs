use crate::gt::*;
use crate::labels::LabelSet;
use crate::mimicgraph::MimicGraphOptions;
use hnsw_itu::Point;
use tracing::info;

pub mod filtered;
pub mod index;

pub struct FilteredMimicGraphOptions {
    pub base_options: MimicGraphOptions,
    pub threshold: usize,
    pub labels: Vec<LabelSet>,
    pub query_labels: Vec<LabelSet>,
}

impl Default for FilteredMimicGraphOptions {
    fn default() -> Self {
        Self {
            base_options: MimicGraphOptions::default(),
            threshold: 1000,
            labels: vec![],
            query_labels: vec![],
        }
    }
}

impl FilteredMimicGraphOptions {
    pub fn tuned<P>(
        data: &[P],
        queries: &[P],
        labels: Vec<LabelSet>,
        query_labels: Vec<LabelSet>,
    ) -> FilteredMimicGraphOptions
    where
        P: Point + Sync,
    {
        info!("Auto tuning options...");

        let k = 100;
        let slots = 10;
        let data_size = data.len().min(data.len().clamp(10000, 100_000));
        let queries_size = queries
            .len()
            .min(((data_size * slots) as f32 / k as f32).round() as usize);

        let data = &data[..data_size];
        let queries = &queries[..queries_size];
        let labels_sample = &labels[..data_size];
        let query_labels_sample = &query_labels[..queries_size];

        let gt =
            compute_filtered_ground_truth(queries, data, labels_sample, query_labels_sample, k);

        let (data_fraction, data_spread) = fraction_and_spread(&gt, data_size, k);

        info!(data_fraction, data_spread, data_size, queries_size);

        let options = Self::from_fraction_and_spread(data_fraction, data_spread);

        info!(
            ?options.m, ?options.l, ?options.p,
            "Tuned options",
        );

        Self {
            base_options: options,
            labels,
            query_labels,
            ..Default::default()
        }
    }

    fn from_fraction_and_spread(data_fraction: f32, data_spread: f32) -> MimicGraphOptions {
        MimicGraphOptions {
            m: (100.0 - 80.0 * (0.7 * data_fraction + 0.3 * data_spread)).clamp(24.0, 80.0)
                as usize,
            l: (-200.0 + 900.0 * (0.7 * data_fraction + 0.3 * data_spread)).clamp(100.0, 700.0)
                as usize,
            p: (-200.0 + 900.0 * (0.7 * data_fraction + 0.3 * data_spread)).clamp(100.0, 700.0)
                as usize,
            ..Default::default()
        }
    }
}

impl std::fmt::Debug for FilteredMimicGraphOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteredMimicGraphOptions")
            .field("base_options", &self.base_options)
            .field("threshold", &self.threshold)
            .field("labels", &format_args!("<{} items>", self.labels.len()))
            .field(
                "query_labels",
                &format_args!("<{} items>", self.query_labels.len()),
            )
            .finish()
    }
}
