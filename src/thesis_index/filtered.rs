use crate::labels::LabelledPoint;
use hnsw_itu::{Distance, Index, IndexVis, Point};
use roargraph::AdjListGraph;
use std::collections::HashSet;

pub struct FilteredIndex<T> {
    pub(crate) graph: AdjListGraph<LabelledPoint<T>>,
}

impl<P: Point> Index<LabelledPoint<P>> for FilteredIndex<P> {
    type Options = ();

    fn size(&self) -> usize {
        self.graph.size()
    }

    fn search(
        &'_ self,
        query: &LabelledPoint<P>,
        k: usize,
        options: &Self::Options,
    ) -> Vec<Distance<'_, LabelledPoint<P>>>
    where
        P: Point,
    {
        todo!()
    }
}

impl<P: Point> IndexVis<LabelledPoint<P>> for FilteredIndex<P> {
    fn search_vis<'a>(
        &'a self,
        query: &LabelledPoint<P>,
        k: usize,
        options: &Self::Options,
        vis: &mut HashSet<Distance<'a, LabelledPoint<P>>>,
    ) -> Vec<Distance<'a, LabelledPoint<P>>>
    where
        LabelledPoint<P>: Point,
    {
        todo!()
    }
}
