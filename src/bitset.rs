//! Sparse bitset implementation based on hi_sparse_bitset.
//!
//! The implementation maintains a hierarchy where the leaf nodes contain dense bitsets and the
//! internal nodes contain pointers to their children. The hierarchy is built lazily, where the
//! fully saturated bitset will always have the same structure and each data node will always cover
//! the same indices. E.g., if `size_of::<usize>() == 8` and bits 1 and 101 are set, there will be
//! two data blocks covering [0, 64) and [64, 128), respectively. Data block coverage ranges will
//! always be aligned to `size_of::<usize>() * 8`.

use std::ops::{BitAnd, BitOr};

pub struct Bitset {
    top: Block,
    /// Possible range of indices in the bitset without resizing
    /// [inclusive, exclusive)
    bounds: (usize, usize),
    /// Actual range of indices in the bitset
    /// [inclusive, exclusive)
    range: (usize, usize),
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
        todo!()
    }

    /// Sets the bit at the given index
    pub fn set(&mut self, index: usize) {
        todo!()
    }

    /// Clears the bit at the given index
    pub fn clear(&mut self, index: usize) {
        todo!()
    }

    /// Returns true if the bitset is empty
    pub fn is_empty(&self) -> bool {
        matches!(self.top, Block::Empty)
    }

    /// Returns the number of bits set in the bitset
    pub fn count(&self) -> usize {
        todo!()
    }
}

/// Intersection of two bitsets
impl BitAnd for &Bitset {
    type Output = Bitset;

    fn bitand(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

/// Union of two bitsets
impl BitOr for &Bitset {
    type Output = Bitset;

    fn bitor(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

type BlockIndex = usize;

enum Block {
    Empty,
    Data(BlockIndex),
    Level(BlockIndex, Box<[Block; size_of::<BlockIndex>()]>),
}
