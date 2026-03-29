use crate::vamana::LabelledPoint;
use crate::vamana::index::FilteredVamana;
use hnsw_itu::{IndexBuilder, Point};
use roargraph::AdjListGraph;

pub struct StitchedVamanaBuilder<P> {
    graph: AdjListGraph<P>,
}

impl<P: Point> IndexBuilder<LabelledPoint<P>> for StitchedVamanaBuilder<LabelledPoint<P>> {
    type Index = FilteredVamana<LabelledPoint<P>>;

    fn add(&mut self, point: LabelledPoint<P>) {
        todo!()
    }

    fn build(self) -> Self::Index {
        todo!()
    }
}
