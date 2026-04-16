//! Sparse bitset implementation based on hi_sparse_bitset.
//!
//! The implementation maintains a hierarchy where the leaf nodes contain dense bitsets and the
//! internal nodes contain pointers to their children. The hierarchy is built lazily, where the
//! fully saturated bitset will always have the same structure and each data node will always cover
//! the same indices. E.g., if `size_of::<usize>() == 8` and bits 1 and 101 are set, there will be
//! two data blocks covering [0, 64) and [64, 128), respectively. Data block coverage ranges will
//! always be aligned to `size_of::<usize>() * 8`.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Sub, SubAssign};

type BlockIndex = usize;

/// Number of bits stored in a single Data block.
const BITS_PER_DATA: usize = BlockIndex::BITS as usize;

/// Number of children per Level block.
const BRANCHING: usize = size_of::<BlockIndex>();

#[derive(Clone)]
pub struct Bitset {
    top: Block,
    /// Possible range of indices in the bitset without resizing
    /// [inclusive, exclusive)
    bounds: (usize, usize),
    /// Actual range of indices in the bitset
    /// [inclusive, exclusive)
    range: (usize, usize),
}

impl Serialize for Bitset {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bits: Vec<usize> = self.iter().collect();

        bits.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Bitset {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bits: Vec<usize> = Vec::deserialize(deserializer)?;
        let mut bitset = Bitset::new();

        for index in bits {
            bitset.set(index);
        }

        Ok(bitset)
    }
}

impl Default for Bitset {
    fn default() -> Self {
        Self {
            top: Block::Empty,
            bounds: (0, 0),
            range: (0, 0),
        }
    }
}

impl Bitset {
    /// Create an empty bitset
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if the bit at the given index is set
    pub fn is_set(&self, index: usize) -> bool {
        if index < self.range.0 || index >= self.range.1 {
            return false;
        }

        self.top.is_set(index, self.bounds.0, self.bounds.1)
    }

    /// Sets the bit at the given index. Returns `true` if the bit was newly set.
    pub fn set(&mut self, index: usize) -> bool {
        if self.is_empty() {
            let block_start = (index / BITS_PER_DATA) * BITS_PER_DATA;
            self.bounds = (block_start, block_start + BITS_PER_DATA);
            self.range = (index, index + 1);
        }

        while index < self.bounds.0 || index >= self.bounds.1 {
            self.grow();
        }

        if index < self.range.0 {
            self.range.0 = index;
        }
        if index >= self.range.1 {
            self.range.1 = index + 1;
        }

        self.top.set(index, self.bounds.0, self.bounds.1)
    }

    /// Alias for [`Bitset::set`]. Returns `true` if the bit was newly set.
    pub fn insert(&mut self, index: usize) -> bool {
        self.set(index)
    }

    /// Clears the bit at the given index
    pub fn clear(&mut self, index: usize) {
        if index < self.range.0 || index >= self.range.1 {
            return;
        }

        self.top.clear(index, self.bounds.0, self.bounds.1);

        if self.top.is_empty() {
            self.range = (0, 0);
            self.bounds = (0, 0);
        } else {
            if index == self.range.0 {
                self.range.0 = self.top.min_bit(self.bounds.0, self.bounds.1).unwrap();
            } else if index == self.range.1 - 1 {
                self.range.1 = self.top.max_bit(self.bounds.0, self.bounds.1).unwrap() + 1;
            }

            self.shrink();
        }
    }

    /// Returns the smallest set bit index, or `None` if empty.
    pub fn min(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.range.0)
        }
    }

    /// Returns the largest set bit index, or `None` if empty.
    pub fn max(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.range.1 - 1)
        }
    }

    /// Returns true if the bitset is empty
    pub fn is_empty(&self) -> bool {
        self.top.is_empty()
    }

    /// Returns the number of bits set in the bitset
    pub fn count(&self) -> usize {
        self.top.count()
    }

    /// Returns an iterator over the indices of all set bits.
    pub fn iter(&self) -> BitsetIter<'_> {
        BitsetIter {
            stack: vec![IterFrame::new(&self.top, self.bounds.0, self.bounds.1)],
            data_bits: 0,
            data_start: 0,
        }
    }

    /// Grows the hierarchy by one level, wrapping the current top block as a child.
    fn grow(&mut self) {
        let coverage = self.bounds.1 - self.bounds.0;
        let new_coverage = coverage * BRANCHING;
        let new_start = (self.bounds.0 / new_coverage) * new_coverage;
        let child_idx = (self.bounds.0 - new_start) / coverage;

        let old_top = std::mem::replace(&mut self.top, Block::Empty);
        let mask = if old_top.is_empty() {
            0
        } else {
            1usize << child_idx
        };

        let mut children: [Block; BRANCHING] = std::array::from_fn(|_| Block::Empty);
        children[child_idx] = old_top;

        self.top = Block::Level(mask, Box::new(children));
        self.bounds = (new_start, new_start + new_coverage);
    }

    /// Shrinks the hierarchy by removing unnecessary top-level wrappers.
    /// If the top is a Level with exactly one non-empty child, unwrap it.
    fn shrink(&mut self) {
        loop {
            match &mut self.top {
                Block::Level(mask, children) if mask.count_ones() == 1 => {
                    let i = mask.trailing_zeros() as usize;
                    let child_coverage = (self.bounds.1 - self.bounds.0) / BRANCHING;
                    let child_start = self.bounds.0 + i * child_coverage;

                    self.top = std::mem::replace(&mut children[i], Block::Empty);
                    self.bounds = (child_start, child_start + child_coverage);
                }
                _ => break,
            }
        }
    }
}

/// Iterator over set bit indices in a [`Bitset`].
pub struct BitsetIter<'a> {
    stack: Vec<IterFrame<'a>>,
    data_bits: BlockIndex,
    data_start: usize,
}

struct IterFrame<'a> {
    block: &'a Block,
    start: usize,
    end: usize,
    child: usize,
}

impl<'a> IterFrame<'a> {
    fn new(block: &'a Block, start: usize, end: usize) -> Self {
        Self {
            block,
            start,
            end,
            child: 0,
        }
    }
}

impl<'a> Iterator for BitsetIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            // Drain any remaining bits from the current data block
            if self.data_bits != 0 {
                let bit = self.data_bits.trailing_zeros() as usize;
                self.data_bits &= self.data_bits - 1;

                return Some(self.data_start + bit);
            }

            // Walk the tree to find the next data block
            let frame = self.stack.last_mut()?;
            match frame.block {
                Block::Empty => {
                    self.stack.pop();
                }
                Block::Data(bits) => {
                    self.data_bits = *bits;
                    self.data_start = frame.start;
                    self.stack.pop();
                }
                Block::Level(mask, children) => {
                    let child_coverage = (frame.end - frame.start) / BRANCHING;
                    let start = frame.start;
                    let mask = *mask;

                    // Find the next non-empty child
                    let mut found = None;
                    while frame.child < BRANCHING {
                        let i = frame.child;
                        frame.child += 1;
                        if (mask >> i) & 1 == 1 {
                            found = Some((i, &children[i]));
                            break;
                        }
                    }

                    if frame.child >= BRANCHING {
                        self.stack.pop();
                    }

                    if let Some((i, child)) = found {
                        let child_start = start + i * child_coverage;
                        self.stack.push(IterFrame::new(
                            child,
                            child_start,
                            child_start + child_coverage,
                        ));
                    }
                }
            }
        }
    }
}

impl<'a> IntoIterator for &'a Bitset {
    type Item = usize;
    type IntoIter = BitsetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Extend<usize> for Bitset {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, iter: I) {
        for index in iter {
            self.set(index);
        }
    }
}

/// Intersection of two bitsets
impl BitAnd for &Bitset {
    type Output = Bitset;

    fn bitand(self, rhs: Self) -> Self::Output {
        if self.range.0 >= rhs.range.1 || rhs.range.0 >= self.range.1 {
            return Bitset::new();
        }

        let mut result = Bitset::new();
        for index in self.iter() {
            if rhs.is_set(index) {
                result.set(index);
            }
        }

        result
    }
}

/// Union of two bitsets
impl BitOr for &Bitset {
    type Output = Bitset;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mut result = Bitset::new();
        result.extend(self.iter());
        result.extend(rhs.iter());

        result
    }
}

impl BitAndAssign<&Bitset> for Bitset {
    fn bitand_assign(&mut self, rhs: &Bitset) {
        *self = &*self & rhs;
    }
}

impl BitOrAssign<&Bitset> for Bitset {
    fn bitor_assign(&mut self, rhs: &Bitset) {
        self.extend(rhs.iter());
    }
}

impl SubAssign<&Bitset> for Bitset {
    fn sub_assign(&mut self, rhs: &Bitset) {
        *self = &*self - rhs;
    }
}

/// Set difference
impl Sub for &Bitset {
    type Output = Bitset;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = Bitset::new();

        for index in self.iter() {
            if !rhs.is_set(index) {
                result.set(index);
            }
        }

        result
    }
}

impl Sub for Bitset {
    type Output = Bitset;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Sub<&Bitset> for Bitset {
    type Output = Bitset;

    fn sub(self, rhs: &Bitset) -> Self::Output {
        &self - rhs
    }
}

impl Sub<Bitset> for &Bitset {
    type Output = Bitset;

    fn sub(self, rhs: Bitset) -> Self::Output {
        self - &rhs
    }
}

enum Block {
    Empty,
    Data(BlockIndex),
    Level(BlockIndex, Box<[Block; BRANCHING]>),
}

impl Clone for Block {
    fn clone(&self) -> Self {
        match self {
            Block::Empty => Block::Empty,
            Block::Data(bits) => Block::Data(*bits),
            Block::Level(mask, children) => Block::Level(
                *mask,
                Box::new(std::array::from_fn(|i| children[i].clone())),
            ),
        }
    }
}

impl Block {
    fn is_empty(&self) -> bool {
        matches!(self, Block::Empty)
    }

    /// Returns `(child_coverage, child_idx, child_start)` for a given index within `[start, end)`.
    #[inline]
    fn child_of(index: usize, start: usize, end: usize) -> (usize, usize, usize) {
        let child_coverage = (end - start) / BRANCHING;
        let child_idx = (index - start) / child_coverage;
        let child_start = start + child_idx * child_coverage;

        (child_coverage, child_idx, child_start)
    }

    /// Returns `(child_coverage, child_start)` for a known child index within `[start, end)`.
    #[inline]
    fn child_range(i: usize, start: usize, end: usize) -> (usize, usize) {
        let child_coverage = (end - start) / BRANCHING;

        (child_coverage, start + i * child_coverage)
    }

    /// Returns true if the bit at `index` is set within this block covering `[start, end)`.
    fn is_set(&self, index: usize, start: usize, end: usize) -> bool {
        match self {
            Block::Empty => false,
            Block::Data(bits) => (bits >> (index - start)) & 1 == 1,
            Block::Level(mask, children) => {
                let (child_coverage, child_idx, child_start) = Self::child_of(index, start, end);

                if (mask >> child_idx) & 1 == 0 {
                    return false;
                }

                children[child_idx].is_set(index, child_start, child_start + child_coverage)
            }
        }
    }

    /// Sets the bit at `index` within this block covering `[start, end)`.
    /// Creates child blocks as needed. Returns `true` if the bit was newly set.
    fn set(&mut self, index: usize, start: usize, end: usize) -> bool {
        let coverage = end - start;
        match self {
            Block::Empty => {
                if coverage == BITS_PER_DATA {
                    *self = Block::Data(1 << (index - start));
                } else {
                    let (_, child_idx, child_start) = Self::child_of(index, start, end);
                    let mut children: [Block; BRANCHING] = std::array::from_fn(|_| Block::Empty);

                    children[child_idx].set(index, child_start, child_start + coverage / BRANCHING);

                    *self = Block::Level(1 << child_idx, Box::new(children));
                }

                true
            }
            Block::Data(bits) => {
                let bit = 1 << (index - start);
                let was_new = *bits & bit == 0;
                *bits |= bit;

                was_new
            }
            Block::Level(mask, children) => {
                let (child_coverage, child_idx, child_start) = Self::child_of(index, start, end);
                *mask |= 1 << child_idx;

                children[child_idx].set(index, child_start, child_start + child_coverage)
            }
        }
    }

    /// Clears the bit at `index` within this block covering `[start, end)`.
    /// Collapses empty blocks.
    fn clear(&mut self, index: usize, start: usize, end: usize) {
        match self {
            Block::Empty => {}
            Block::Data(bits) => {
                *bits &= !(1 << (index - start));

                if *bits == 0 {
                    *self = Block::Empty;
                }
            }
            Block::Level(mask, children) => {
                let (child_coverage, child_idx, child_start) = Self::child_of(index, start, end);

                if ((*mask >> child_idx) & 1) == 0 {
                    return;
                }

                children[child_idx].clear(index, child_start, child_start + child_coverage);
                if children[child_idx].is_empty() {
                    *mask &= !(1 << child_idx);
                    if *mask == 0 {
                        *self = Block::Empty;
                    }
                }
            }
        }
    }

    /// Returns the number of bits set in this block.
    fn count(&self) -> usize {
        match self {
            Block::Empty => 0,
            Block::Data(bits) => bits.count_ones() as usize,
            Block::Level(_, children) => children.iter().map(|c| c.count()).sum(),
        }
    }

    /// Returns the smallest set bit index in this block covering `[start, end)`.
    fn min_bit(&self, start: usize, end: usize) -> Option<usize> {
        match self {
            Block::Empty => None,
            Block::Data(bits) => Some(start + bits.trailing_zeros() as usize),
            Block::Level(mask, children) => {
                let i = mask.trailing_zeros() as usize;
                let (child_coverage, child_start) = Self::child_range(i, start, end);

                children[i].min_bit(child_start, child_start + child_coverage)
            }
        }
    }

    /// Returns the largest set bit index in this block covering `[start, end)`.
    fn max_bit(&self, start: usize, end: usize) -> Option<usize> {
        match self {
            Block::Empty => None,
            Block::Data(bits) => {
                Some(start + (BlockIndex::BITS - 1 - bits.leading_zeros()) as usize)
            }
            Block::Level(mask, children) => {
                let i = (BlockIndex::BITS - 1 - mask.leading_zeros()) as usize;
                let (child_coverage, child_start) = Self::child_range(i, start, end);

                children[i].max_bit(child_start, child_start + child_coverage)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_query() {
        let mut b = Bitset::new();
        b.extend([0, 42, 100]);

        assert!(b.is_set(0));
        assert!(b.is_set(42));
        assert!(b.is_set(100));
        assert!(!b.is_set(1));
        assert!(!b.is_set(99));
        assert_eq!(b.count(), 3);
    }

    #[test]
    fn intersection_union_difference() {
        let mut a = Bitset::new();
        let mut b = Bitset::new();
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
    fn iter_yields_sorted_indices() {
        let mut b = Bitset::new();

        // Insert in arbitrary order, including across block boundaries
        b.extend([500, 3, 64, 0, 128, 65, 1000]);

        let items: Vec<usize> = b.iter().collect();
        assert_eq!(items, vec![0, 3, 64, 65, 128, 500, 1000]);
    }

    #[test]
    fn empty_bitset() {
        let b = Bitset::new();

        assert!(b.is_empty());
        assert_eq!(b.count(), 0);
        assert!(!b.is_set(0));
        assert_eq!(b.iter().count(), 0);
    }

    #[test]
    fn clear_last_bit_makes_empty() {
        let mut b = Bitset::new();

        b.set(7);
        assert!(!b.is_empty());

        b.clear(7);
        assert!(b.is_empty());
        assert_eq!(b.count(), 0);
        assert_eq!(b.min(), None);
        assert_eq!(b.max(), None);

        b.set(100);
        assert!(b.is_set(100));
        assert_eq!(b.count(), 1);
    }

    #[test]
    fn set_same_bit_twice_is_idempotent() {
        let mut b = Bitset::new();

        b.set(42);
        b.set(42);

        assert_eq!(b.count(), 1);
    }

    #[test]
    fn set_returns_true_only_on_new_bit() {
        let mut b = Bitset::new();

        assert!(b.set(10));
        assert!(!b.set(10));
        assert!(b.insert(200));
        assert!(!b.insert(200));
        assert!(b.set(11));
        assert_eq!(b.count(), 3);
    }

    #[test]
    fn clear_unset_bit_is_noop() {
        let mut b = Bitset::new();

        b.set(10);
        b.clear(999);
        b.clear(5);

        assert_eq!(b.count(), 1);
        assert!(b.is_set(10));
    }

    #[test]
    fn clear_shrinks_range_lower_bound() {
        let mut b = Bitset::new();
        b.extend([10, 50, 200]);

        assert_eq!(b.min(), Some(10));
        b.clear(10);
        assert_eq!(b.min(), Some(50));
        assert!(b.is_set(50));
        assert!(b.is_set(200));
        assert!(!b.is_set(10));
    }

    #[test]
    fn clear_shrinks_range_upper_bound() {
        let mut b = Bitset::new();
        b.extend([10, 50, 200]);

        assert_eq!(b.max(), Some(200));
        b.clear(200);
        assert_eq!(b.max(), Some(50));
        assert!(b.is_set(10));
        assert!(b.is_set(50));
        assert!(!b.is_set(200));
    }

    #[test]
    fn clear_both_bounds_sequentially() {
        let mut b = Bitset::new();
        b.extend([5, 100, 1000]);

        b.clear(5);
        assert_eq!(b.min(), Some(100));
        b.clear(1000);
        assert_eq!(b.max(), Some(100));
        assert_eq!(b.count(), 1);
        assert!(b.is_set(100));
    }

    #[test]
    fn clear_middle_bit_does_not_change_range() {
        let mut b = Bitset::new();
        b.extend([10, 50, 200]);

        let min_before = b.min();
        let max_before = b.max();
        b.clear(50);

        assert_eq!(b.min(), min_before);
        assert_eq!(b.max(), max_before);
        assert_eq!(b.count(), 2);
    }

    #[test]
    fn clear_shrinks_bounds() {
        let mut b = Bitset::new();
        b.extend([10, 50, 10_000]);
        let bounds_wide = b.bounds;
        b.clear(10_000);

        // Bounds should have shrunk since all remaining bits fit in a smaller tree
        assert!(
            b.bounds.1 - b.bounds.0 < bounds_wide.1 - bounds_wide.0,
            "bounds should shrink: was {:?}, now {:?}",
            bounds_wide,
            b.bounds
        );

        assert!(b.is_set(10));
        assert!(b.is_set(50));
        assert!(!b.is_set(10_000));
        assert_eq!(b.count(), 2);
    }

    #[test]
    fn intersection_disjoint_ranges() {
        let mut a = Bitset::new();
        let mut b = Bitset::new();
        a.extend([0, 1]);
        b.extend([1000, 1001]);

        let inter = &a & &b;
        assert!(inter.is_empty());
    }

    #[test]
    fn clone_is_independent() {
        let mut a = Bitset::new();
        a.extend([5, 200]);
        let mut b = a.clone();
        b.set(99);
        b.clear(5);

        assert!(a.is_set(5));
        assert!(!a.is_set(99));
        assert!(!b.is_set(5));
        assert!(b.is_set(99));
    }

    #[test]
    fn serde_roundtrip() {
        let mut b = Bitset::new();
        b.extend([0, 63, 64, 127, 1000]);
        let json = serde_json::to_string(&b).unwrap();
        let b2: Bitset = serde_json::from_str(&json).unwrap();
        assert_eq!(b.iter().collect::<Vec<_>>(), b2.iter().collect::<Vec<_>>());
    }

    #[test]
    fn stress_set_clear_many_bits() {
        use std::collections::BTreeSet;

        let mut bitset = Bitset::new();
        let mut reference = BTreeSet::new();

        // Insert a spread of indices
        let indices: Vec<usize> = (0..500).map(|i| i * 37 % 10_000).collect();
        for &i in &indices {
            let new_ref = reference.insert(i);
            let new_bs = bitset.set(i);
            assert_eq!(new_bs, new_ref, "set({i}) disagreed on novelty");
        }
        assert_eq!(bitset.count(), reference.len());
        assert_eq!(bitset.min(), reference.iter().next().copied(),);
        assert_eq!(bitset.max(), reference.iter().next_back().copied(),);

        // Remove every other inserted index
        for &i in indices.iter().step_by(2) {
            reference.remove(&i);
            bitset.clear(i);
        }
        assert_eq!(bitset.count(), reference.len());
        let bs_vec: Vec<usize> = bitset.iter().collect();
        let ref_vec: Vec<usize> = reference.iter().copied().collect();
        assert_eq!(bs_vec, ref_vec, "iteration mismatch after removals");
        assert_eq!(bitset.min(), reference.iter().next().copied());
        assert_eq!(bitset.max(), reference.iter().next_back().copied());

        for i in 0..10_000 {
            assert_eq!(
                bitset.is_set(i),
                reference.contains(&i),
                "is_set({i}) mismatch"
            );
        }

        // Clear everything and verify
        let remaining: Vec<usize> = reference.iter().copied().collect();
        for &i in &remaining {
            bitset.clear(i);
            reference.remove(&i);
            assert_eq!(bitset.count(), reference.len(), "count after clearing {i}");
            assert_eq!(
                bitset.min(),
                reference.iter().next().copied(),
                "min after clearing {i}"
            );
            assert_eq!(
                bitset.max(),
                reference.iter().next_back().copied(),
                "max after clearing {i}"
            );
        }
        assert!(bitset.is_empty());

        // Reuse after clear
        bitset.set(42);
        assert!(bitset.is_set(42));
        assert_eq!(bitset.count(), 1);
    }

    #[test]
    fn stress_wide_spread_indices() {
        use std::collections::BTreeSet;

        let mut bitset = Bitset::new();
        let mut reference = BTreeSet::new();

        // Indices spread across a very wide range
        let indices: Vec<usize> = (0..200).map(|i| (i * 4999 + 7) % 1_000_000).collect();

        for &i in &indices {
            reference.insert(i);
            bitset.set(i);
        }
        assert_eq!(bitset.count(), reference.len());
        assert_eq!(
            bitset.iter().collect::<Vec<_>>(),
            reference.iter().copied().collect::<Vec<_>>(),
        );

        // Clear from the max end
        while let Some(&max) = reference.iter().next_back() {
            bitset.clear(max);
            reference.remove(&max);
            assert_eq!(bitset.max(), reference.iter().next_back().copied());
            assert_eq!(bitset.count(), reference.len());
        }
        assert!(bitset.is_empty());
    }

    #[test]
    fn stress_clear_from_min_end() {
        use std::collections::BTreeSet;

        let mut bitset = Bitset::new();
        let mut reference = BTreeSet::new();

        for i in 0..64 {
            reference.insert(i);
            bitset.set(i);
        }
        for &i in &[500, 5_000, 50_000] {
            reference.insert(i);
            bitset.set(i);
        }

        // Clear from the min end
        while let Some(&min) = reference.iter().next() {
            bitset.clear(min);
            reference.remove(&min);
            assert_eq!(
                bitset.min(),
                reference.iter().next().copied(),
                "min after clearing {min}"
            );
            assert_eq!(
                bitset.count(),
                reference.len(),
                "count after clearing {min}"
            );
        }
        assert!(bitset.is_empty());
    }

    #[test]
    fn stress_interleaved_set_clear() {
        use std::collections::BTreeSet;

        let mut bitset = Bitset::new();
        let mut reference = BTreeSet::new();

        // Interleave insertions and deletions
        for round in 0..10 {
            for j in 0..50 {
                let idx = round * 1000 + j * 13;
                reference.insert(idx);
                bitset.set(idx);
            }

            let to_remove: Vec<usize> = reference
                .iter()
                .copied()
                .filter(|&x| x % (round + 2) == 0)
                .collect();
            for idx in to_remove {
                reference.remove(&idx);
                bitset.clear(idx);
            }

            assert_eq!(
                bitset.count(),
                reference.len(),
                "count mismatch at round {round}"
            );
            assert_eq!(
                bitset.min(),
                reference.iter().next().copied(),
                "min mismatch at round {round}"
            );
            assert_eq!(
                bitset.max(),
                reference.iter().next_back().copied(),
                "max mismatch at round {round}"
            );
            assert_eq!(
                bitset.iter().collect::<Vec<_>>(),
                reference.iter().copied().collect::<Vec<_>>(),
                "iter mismatch at round {round}"
            );
        }
    }

    #[test]
    fn stress_set_operations_after_clear() {
        let mut a = Bitset::new();
        let mut b = Bitset::new();

        for i in (0..1000).step_by(3) {
            a.set(i);
        }
        for i in (0..1000).step_by(5) {
            b.set(i);
        }

        a.clear(0);
        a.clear(999);

        let inter = &a & &b;
        let union = &a | &b;
        let diff = &a - &b;

        for i in 0..1000 {
            let in_a = i % 3 == 0 && i != 0 && i != 999;
            let in_b = i % 5 == 0;
            assert_eq!(inter.is_set(i), in_a && in_b, "intersection at {i}");
            assert_eq!(union.is_set(i), in_a || in_b, "union at {i}");
            assert_eq!(diff.is_set(i), in_a && !in_b, "difference at {i}");
        }
    }
}
