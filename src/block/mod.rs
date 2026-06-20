//! SIMD [`Block`] abstraction (Section 4, BlockWrapper in the reference).

mod repeat;

#[cfg(any(
    halftime_backend = "soft",
    all(not(halftime_backend = "soft"), not(target_arch = "aarch64")),
))]
mod scalar;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
mod x86;

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
mod neon;

pub(crate) use repeat::RepeatBlock;

#[cfg(any(
    halftime_backend = "soft",
    all(not(halftime_backend = "soft"), not(target_arch = "aarch64")),
))]
pub(crate) use scalar::ScalarBlock;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) use x86::{Avx2Repeat2Block, Avx512Block, Sse2Block};

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) use neon::NeonBlock;

/// Lane-wise 64-bit block used throughout EHC / tree / finalizer.
pub(crate) trait Block: Copy + Clone {
    /// Parallel 64-bit lanes (`b` in the paper).
    const LANES: usize;
    /// Stored width in bytes (`sizeof(Block)`).
    const BYTES: usize;

    /// Zero block.
    fn zero() -> Self {
        Self::load_one(0)
    }

    fn load(ptr: *const u8) -> Self;
    fn load_one(word: u64) -> Self;
    fn plus(self, rhs: Self) -> Self;
    fn plus32(self, rhs: Self) -> Self;
    fn times(self, rhs: Self) -> Self;
    fn xor(self, rhs: Self) -> Self;
    /// Left shift by a compile-time bit count (SIMD immediates).
    fn shl<const BITS: u32>(self) -> Self;
    fn minus(self, rhs: Self) -> Self;
    fn right_shift32(self) -> Self;
    /// Reduce SIMD lanes to one 64-bit word (identity for scalar).
    fn sum(self) -> u64;
}

#[inline(always)]
pub(crate) fn multiply_add<B: Block>(summand: B, factor1: B, factor2: B) -> B {
    summand.plus(factor1.times(factor2))
}
