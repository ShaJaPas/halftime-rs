//! Scalar block (`b = 1`, V1 in the reference).

use super::Block;

/// Single-lane block (`uint64_t` in the reference).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ScalarBlock(pub u64);

impl Block for ScalarBlock {
    const LANES: usize = 1;
    const BYTES: usize = 8;

    #[inline(always)]
    fn load(ptr: *const u8) -> Self {
        let mut word = 0u64;
        // SAFETY: See function safety invariants; operands satisfy `Block` / pointer contracts.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr, &mut word as *mut u64 as *mut u8, 8);
        }
        Self(word)
    }

    #[inline(always)]
    fn load_one(word: u64) -> Self {
        Self(word)
    }

    #[inline(always)]
    fn plus(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }

    #[inline(always)]
    fn plus32(self, rhs: Self) -> Self {
        let lo = (self.0 as u32).wrapping_add(rhs.0 as u32) as u64;
        let hi = (self.0 >> 32).wrapping_add(rhs.0 >> 32) << 32;
        Self(lo | hi)
    }

    #[inline(always)]
    fn times(self, rhs: Self) -> Self {
        let mask = (1u64 << 32) - 1;
        Self((self.0 & mask).wrapping_mul(rhs.0 & mask))
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }

    #[inline(always)]
    fn shl<const BITS: u32>(self) -> Self {
        Self(self.0.wrapping_shl(BITS))
    }

    #[inline(always)]
    fn minus(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }

    #[inline(always)]
    fn right_shift32(self) -> Self {
        Self(self.0 >> 32)
    }

    #[inline(always)]
    fn sum(self) -> u64 {
        self.0
    }
}
