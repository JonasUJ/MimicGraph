//! Sparse bitset implementation based on hi_sparse_bitset.
//!
//! The implementation maintains a hierarchy where the leaf nodes contain dense bitsets and the
//! internal nodes contain pointers to their children. The hierarchy is built lazily, where the
//! fully saturated bitset will always have the same structure and each data node will always cover
//! the same indices. E.g., if `size_of::<usize>() == 8` and bits 1 and 101 are set, there will be
//! two data blocks covering [0, 64) and [64, 128), respectively. Data block coverage ranges will
//! always be aligned to `size_of::<usize>() * 8`.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::{BitAnd, BitOr, Sub};

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
    /// Create a new bitset with no bits set
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

    /// Sets the bit at the given index
    pub fn set(&mut self, index: usize) {
        if self.is_empty() {
            let block_start = (index / BITS_PER_DATA) * BITS_PER_DATA;
            self.bounds = (block_start, block_start + BITS_PER_DATA);
            self.range = (index, index + 1);
        }
        while index < self.bounds.0 || index >= self.bounds.1 {
            self.grow();
        }
        self.top.set(index, self.bounds.0, self.bounds.1);
        if index < self.range.0 {
            self.range.0 = index;
        }
        if index >= self.range.1 {
            self.range.1 = index + 1;
        }
    }

    /// Alias for [`Bitset::set`].
    pub fn insert(&mut self, index: usize) {
        self.set(index);
    }

    /// Clears the bit at the given index
    pub fn clear(&mut self, index: usize) {
        if index < self.range.0 || index >= self.range.1 {
            return;
        }
        self.top.clear(index, self.bounds.0, self.bounds.1);
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
        self.top
            .for_each_bit(self.bounds.0, self.bounds.1, &mut |index| {
                if rhs.is_set(index) {
                    result.set(index);
                }
            });
        result
    }
}

/// Union of two bitsets
impl BitOr for &Bitset {
    type Output = Bitset;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mut result = Bitset::new();
        self.top
            .for_each_bit(self.bounds.0, self.bounds.1, &mut |index| {
                result.set(index);
            });
        rhs.top
            .for_each_bit(rhs.bounds.0, rhs.bounds.1, &mut |index| {
                result.set(index);
            });
        result
    }
}

/// Set difference
impl Sub for &Bitset {
    type Output = Bitset;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = Bitset::new();
        self.top
            .for_each_bit(self.bounds.0, self.bounds.1, &mut |index| {
                if !rhs.is_set(index) {
                    result.set(index);
                }
            });
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

    /// Returns true if the bit at `index` is set within this block covering `[start, end)`.
    fn is_set(&self, index: usize, start: usize, end: usize) -> bool {
        match self {
            Block::Empty => false,
            Block::Data(bits) => (bits >> (index - start)) & 1 == 1,
            Block::Level(mask, children) => {
                let child_coverage = (end - start) / BRANCHING;
                let child_idx = (index - start) / child_coverage;
                if (mask >> child_idx) & 1 == 0 {
                    return false;
                }
                let child_start = start + child_idx * child_coverage;
                children[child_idx].is_set(index, child_start, child_start + child_coverage)
            }
        }
    }

    /// Sets the bit at `index` within this block covering `[start, end)`.
    /// Creates child blocks as needed.
    fn set(&mut self, index: usize, start: usize, end: usize) {
        let coverage = end - start;
        match self {
            Block::Empty => {
                if coverage == BITS_PER_DATA {
                    *self = Block::Data(1 << (index - start));
                } else {
                    let child_coverage = coverage / BRANCHING;
                    let child_idx = (index - start) / child_coverage;
                    let child_start = start + child_idx * child_coverage;
                    let mut children: [Block; BRANCHING] = std::array::from_fn(|_| Block::Empty);
                    children[child_idx].set(index, child_start, child_start + child_coverage);
                    *self = Block::Level(1 << child_idx, Box::new(children));
                }
            }
            Block::Data(bits) => {
                *bits |= 1 << (index - start);
            }
            Block::Level(mask, children) => {
                let child_coverage = coverage / BRANCHING;
                let child_idx = (index - start) / child_coverage;
                let child_start = start + child_idx * child_coverage;
                children[child_idx].set(index, child_start, child_start + child_coverage);
                *mask |= 1 << child_idx;
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
                let child_coverage = (end - start) / BRANCHING;
                let child_idx = (index - start) / child_coverage;
                if ((*mask >> child_idx) & 1) == 0 {
                    return;
                }
                let child_start = start + child_idx * child_coverage;
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

    /// Calls `f` for each set bit index in this block covering `[start, end)`.
    fn for_each_bit(&self, start: usize, end: usize, f: &mut impl FnMut(usize)) {
        match self {
            Block::Empty => {}
            Block::Data(bits) => {
                let mut remaining = *bits;
                while remaining != 0 {
                    let bit = remaining.trailing_zeros() as usize;
                    f(start + bit);
                    remaining &= remaining - 1;
                }
            }
            Block::Level(mask, children) => {
                let child_coverage = (end - start) / BRANCHING;
                let mut remaining = *mask;
                while remaining != 0 {
                    let i = remaining.trailing_zeros() as usize;
                    let child_start = start + i * child_coverage;
                    children[i].for_each_bit(child_start, child_start + child_coverage, f);
                    remaining &= remaining - 1;
                }
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
        // Insert in arbitrary order, including across data-block boundaries
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
    }

    #[test]
    fn set_same_bit_twice_is_idempotent() {
        let mut b = Bitset::new();
        b.set(42);
        b.set(42);
        assert_eq!(b.count(), 1);
    }

    #[test]
    fn clear_unset_bit_is_noop() {
        let mut b = Bitset::new();
        b.set(10);
        b.clear(999); // out of bounds — noop
        b.clear(5); // in bounds but not set — noop
        assert_eq!(b.count(), 1);
        assert!(b.is_set(10));
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
}
