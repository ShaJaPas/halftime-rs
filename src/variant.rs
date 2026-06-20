//! HalftimeHash variant parameters (Section 4).

use universal_hash::array::ArraySize;

/// Parameters shared by each [`HalftimeHash`](crate::hash) variant.
#[allow(dead_code)]
pub(crate) trait HalftimeVariant: Copy + 'static {
    /// Number of `u64` words in the internal hash output.
    const OUT: usize;
    /// EHC dimension parameter.
    const DIM: usize;
    /// Encoder parameter.
    const ENC: usize;
    /// Tag width in bytes.
    const TAG_BYTES: usize;
    /// Universal-hash block size for this tag.
    type TagSize: ArraySize;
}

/// `HalftimeHash16`: 16-byte tag, distance-2 code.
pub(crate) mod hh16 {
    pub(crate) const TAG_BYTES: usize = 16;
}

/// `HalftimeHash24`: 24-byte tag, distance-3 code.
pub(crate) mod hh24 {
    pub(crate) const TAG_BYTES: usize = 24;
}

/// `HalftimeHash32`: 32-byte tag, distance-4 code.
pub(crate) mod hh32 {
    pub(crate) const TAG_BYTES: usize = 32;
}

/// `HalftimeHash40`: 40-byte tag, distance-5 code.
pub(crate) mod hh40 {
    pub(crate) const TAG_BYTES: usize = 40;
}

/// Marker type for `HalftimeHash16`.
#[derive(Copy, Clone, Debug)]
pub enum Hh16 {}

impl HalftimeVariant for Hh16 {
    const OUT: usize = 2;
    const DIM: usize = 6;
    const ENC: usize = 7;
    const TAG_BYTES: usize = 16;
    type TagSize = universal_hash::consts::U16;
}

/// Marker type for `HalftimeHash24`.
#[derive(Copy, Clone, Debug)]
pub enum Hh24 {}

impl HalftimeVariant for Hh24 {
    const OUT: usize = 3;
    const DIM: usize = 7;
    const ENC: usize = 9;
    const TAG_BYTES: usize = 24;
    type TagSize = universal_hash::consts::U24;
}

/// Marker type for `HalftimeHash32`.
#[derive(Copy, Clone, Debug)]
pub enum Hh32 {}

impl HalftimeVariant for Hh32 {
    const OUT: usize = 4;
    const DIM: usize = 7;
    const ENC: usize = 10;
    const TAG_BYTES: usize = 32;
    type TagSize = universal_hash::consts::U32;
}

/// Marker type for `HalftimeHash40`.
#[derive(Copy, Clone, Debug)]
pub enum Hh40 {}

impl HalftimeVariant for Hh40 {
    const OUT: usize = 5;
    const DIM: usize = 5;
    const ENC: usize = 9;
    const TAG_BYTES: usize = 40;
    type TagSize = universal_hash::consts::U40;
}
