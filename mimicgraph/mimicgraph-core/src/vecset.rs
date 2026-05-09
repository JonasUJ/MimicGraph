//! Compact sorted-vec set for small, sparse label sets.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Sub, SubAssign};

#[derive(Clone, Debug, Default)]
pub struct VecSet {
    /// Sorted, deduplicated elements.
    data: Vec<u32>,
}

impl Serialize for VecSet {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bits: Vec<usize> = self.iter().collect();
        bits.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for VecSet {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bits: Vec<usize> = Vec::deserialize(deserializer)?;
        let mut set = VecSet::new();
        for index in bits {
            set.insert(index);
        }
        Ok(set)
    }
}

impl VecSet {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Sets the bit at the given index. Returns `true` if newly inserted.
    pub fn set(&mut self, index: usize) -> bool {
        self.insert(index)
    }

    /// Inserts an element. Returns `true` if newly inserted.
    pub fn insert(&mut self, index: usize) -> bool {
        let val = index as u32;
        match self.data.binary_search(&val) {
            Ok(_) => false,
            Err(pos) => {
                self.data.insert(pos, val);
                true
            }
        }
    }

    /// Returns true if the element is present.
    pub fn is_set(&self, index: usize) -> bool {
        let val = index as u32;
        self.data.binary_search(&val).is_ok()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn count(&self) -> usize {
        self.data.len()
    }

    pub fn min(&self) -> Option<usize> {
        self.data.first().map(|&v| v as usize)
    }

    pub fn max(&self) -> Option<usize> {
        self.data.last().map(|&v| v as usize)
    }

    /// Removes an element.
    pub fn clear(&mut self, index: usize) {
        let val = index as u32;
        if let Ok(pos) = self.data.binary_search(&val) {
            self.data.remove(pos);
        }
    }

    /// Iterator over elements in ascending order.
    pub fn iter(&self) -> VecSetIter<'_> {
        VecSetIter {
            inner: self.data.iter(),
        }
    }
}

pub struct VecSetIter<'a> {
    inner: std::slice::Iter<'a, u32>,
}

impl Iterator for VecSetIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        self.inner.next().map(|&v| v as usize)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for VecSetIter<'_> {}

impl<'a> IntoIterator for &'a VecSet {
    type Item = usize;
    type IntoIter = VecSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Extend<usize> for VecSet {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, iter: I) {
        for index in iter {
            self.insert(index);
        }
    }
}

/// Intersection
impl BitAnd for &VecSet {
    type Output = VecSet;

    fn bitand(self, rhs: Self) -> VecSet {
        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < self.data.len() && j < rhs.data.len() {
            match self.data[i].cmp(&rhs.data[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result.push(self.data[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        VecSet { data: result }
    }
}

/// Union
impl BitOr for &VecSet {
    type Output = VecSet;

    fn bitor(self, rhs: Self) -> VecSet {
        let mut result = Vec::with_capacity(self.data.len() + rhs.data.len());
        let (mut i, mut j) = (0, 0);
        while i < self.data.len() && j < rhs.data.len() {
            match self.data[i].cmp(&rhs.data[j]) {
                std::cmp::Ordering::Less => {
                    result.push(self.data[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(rhs.data[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    result.push(self.data[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        result.extend_from_slice(&self.data[i..]);
        result.extend_from_slice(&rhs.data[j..]);
        VecSet { data: result }
    }
}

/// Set difference (self - rhs)
impl Sub for &VecSet {
    type Output = VecSet;

    fn sub(self, rhs: Self) -> VecSet {
        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < self.data.len() && j < rhs.data.len() {
            match self.data[i].cmp(&rhs.data[j]) {
                std::cmp::Ordering::Less => {
                    result.push(self.data[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        result.extend_from_slice(&self.data[i..]);
        VecSet { data: result }
    }
}

impl BitAndAssign<&VecSet> for VecSet {
    fn bitand_assign(&mut self, rhs: &VecSet) {
        *self = &*self & rhs;
    }
}

impl BitOrAssign<&VecSet> for VecSet {
    fn bitor_assign(&mut self, rhs: &VecSet) {
        *self = &*self | rhs;
    }
}

impl SubAssign<&VecSet> for VecSet {
    fn sub_assign(&mut self, rhs: &VecSet) {
        *self = &*self - rhs;
    }
}

impl Sub for VecSet {
    type Output = VecSet;
    fn sub(self, rhs: Self) -> VecSet {
        &self - &rhs
    }
}

impl Sub<&VecSet> for VecSet {
    type Output = VecSet;
    fn sub(self, rhs: &VecSet) -> VecSet {
        &self - rhs
    }
}

impl Sub<VecSet> for &VecSet {
    type Output = VecSet;
    fn sub(self, rhs: VecSet) -> VecSet {
        self - &rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_query() {
        let mut s = VecSet::new();
        s.extend([0, 42, 100]);
        assert!(s.is_set(0));
        assert!(s.is_set(42));
        assert!(s.is_set(100));
        assert!(!s.is_set(1));
        assert_eq!(s.count(), 3);
    }

    #[test]
    fn insert_returns_true_only_on_new() {
        let mut s = VecSet::new();
        assert!(s.set(10));
        assert!(!s.set(10));
        assert!(s.insert(200));
        assert!(!s.insert(200));
    }

    #[test]
    fn intersection_union_difference() {
        let mut a = VecSet::new();
        let mut b = VecSet::new();
        a.extend([1, 2, 3, 100]);
        b.extend([2, 3, 4, 200]);

        let inter = &a & &b;
        assert_eq!(inter.iter().collect::<Vec<_>>(), vec![2, 3]);

        let union = &a | &b;
        assert_eq!(union.iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 100, 200]);

        let diff = &a - &b;
        assert_eq!(diff.iter().collect::<Vec<_>>(), vec![1, 100]);
    }

    #[test]
    fn min_max() {
        let mut s = VecSet::new();
        assert_eq!(s.min(), None);
        assert_eq!(s.max(), None);
        s.extend([50, 10, 200]);
        assert_eq!(s.min(), Some(10));
        assert_eq!(s.max(), Some(200));
    }

    #[test]
    fn clear_element() {
        let mut s = VecSet::new();
        s.extend([10, 50, 200]);
        s.clear(50);
        assert!(!s.is_set(50));
        assert_eq!(s.count(), 2);
        s.clear(999); // no-op
        assert_eq!(s.count(), 2);
    }

    #[test]
    fn serde_roundtrip() {
        let mut s = VecSet::new();
        s.extend([0, 63, 64, 127, 1000]);
        let json = serde_json::to_string(&s).unwrap();
        let s2: VecSet = serde_json::from_str(&json).unwrap();
        assert_eq!(s.iter().collect::<Vec<_>>(), s2.iter().collect::<Vec<_>>());
    }
}


