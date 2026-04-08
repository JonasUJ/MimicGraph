use std::collections::{HashMap, HashSet};
use hnsw_itu::Point;
use rand::seq::IteratorRandom;

#[derive(Debug)]
pub struct LabelledPoint<P> {
    pub point: P,
    pub labels: HashSet<usize>,
}

impl<P: Point> Point for LabelledPoint<P> {
    fn distance(&self, other: &Self) -> f32 {
        self.point.distance(&other.point)
    }
}

pub fn find_medoids<P: Point>(
    data: &[LabelledPoint<P>],
    filters: &HashSet<usize>,
    threshold: usize,
) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    let mut counter: HashMap<usize, usize> = HashMap::from_iter(filters.iter().map(|&i| (i, 0)));

    let mut label_map = HashMap::new();

    for (i, p) in data.iter().enumerate() {
        for label in p.labels.iter() {
            label_map.entry(label).or_insert(vec![]).push(i);
        }
    }

    let mut rng = rand::rng();
    for &f in filters {
        let sample = label_map
            .get(&f)
            .unwrap()
            .iter()
            .sample(&mut rng, threshold);
        let (entry, _) = sample
            .iter()
            .map(|&i| (i, counter.get(&f).unwrap()))
            .min_by_key(|(_, c)| *c)
            .unwrap();
        *counter.get_mut(&f).unwrap() += 1;
        map.insert(f, *entry);
    }

    map
}
