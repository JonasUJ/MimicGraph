use crate::vamana::index::FilteredVamana;
use hnsw_itu::{IndexBuilder, Point};
use roargraph::AdjListGraph;

pub struct StitchedVamanaBuilder<P> {
    _graph: AdjListGraph<P>,
}

impl<P: Point> IndexBuilder<P> for StitchedVamanaBuilder<P> {
    type Index = FilteredVamana<P>;

    fn add(&mut self, _point: P) {
        todo!()
    }

    fn build(self) -> Self::Index {
        todo!()
    }
}
