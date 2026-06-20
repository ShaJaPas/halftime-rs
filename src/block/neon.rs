//! AArch64 NEON block (`BlockWrapper128`, `b = 2`).

use core::arch::aarch64::*;

use super::Block;

/// NEON block: two 64-bit lanes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NeonBlock(pub int64x2_t);

impl Block for NeonBlock {
    const LANES: usize = 2;
    const BYTES: usize = 16;

    #[inline(always)]
    fn load(ptr: *const u8) -> Self {
        // SAFETY: `ptr` addresses at least `Self::BYTES` readable bytes (`Block::load` contract).
        Self(unsafe { vld1q_s64(ptr as *const i64) })
    }

    #[inline(always)]
    fn load_one(word: u64) -> Self {
        // SAFETY: Broadcast/zero intrinsic produces a valid vector register.
        Self(unsafe { vdupq_n_s64(word as i64) })
    }

    #[inline(always)]
    fn plus(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { vaddq_s64(self.0, rhs.0) })
    }

    #[inline(always)]
    fn plus32(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe {
            vreinterpretq_s64_s32(vaddq_s32(
                vreinterpretq_s32_s64(self.0),
                vreinterpretq_s32_s64(rhs.0),
            ))
        })
    }

    #[inline(always)]
    fn times(self, rhs: Self) -> Self {
        // SAFETY: See function safety invariants; operands satisfy `Block` / pointer contracts.
        unsafe {
            let a_lo = vmovn_u64(vreinterpretq_u64_s64(self.0));
            let b_lo = vmovn_u64(vreinterpretq_u64_s64(rhs.0));
            Self(vreinterpretq_s64_u64(vmull_u32(a_lo, b_lo)))
        }
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { veorq_s64(self.0, rhs.0) })
    }

    #[inline(always)]
    fn shl<const BITS: u32>(self) -> Self {
        match BITS {
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            1 => Self(unsafe { vshlq_n_s64(self.0, 1) }),
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            2 => Self(unsafe { vshlq_n_s64(self.0, 2) }),
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            3 => Self(unsafe { vshlq_n_s64(self.0, 3) }),
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            32 => Self(unsafe { vshlq_n_s64(self.0, 32) }),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    fn minus(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { vsubq_s64(self.0, rhs.0) })
    }

    #[inline(always)]
    fn right_shift32(self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { vreinterpretq_s64_u64(vshrq_n_u64(vreinterpretq_u64_s64(self.0), 32)) })
    }

    #[inline(always)]
    fn sum(self) -> u64 {
        // SAFETY: See function safety invariants; operands satisfy `Block` / pointer contracts.
        unsafe { (vgetq_lane_s64(self.0, 0) as u64).wrapping_add(vgetq_lane_s64(self.0, 1) as u64) }
    }
}
