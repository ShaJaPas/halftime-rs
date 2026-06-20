//! x86 SIMD blocks (`BlockWrapper512` / `BlockWrapper256` / `BlockWrapper128`).

use core::arch::x86_64::*;

use super::Block;

/// SSE2 block: two 64-bit lanes (`V2Sse2`, `b = 2`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sse2Block(pub __m128i);

impl Block for Sse2Block {
    const LANES: usize = 2;
    const BYTES: usize = 16;

    #[inline(always)]
    fn load(ptr: *const u8) -> Self {
        // SAFETY: `ptr` addresses at least `Self::BYTES` readable bytes (`Block::load` contract).
        Self(unsafe { _mm_loadu_si128(ptr as *const __m128i) })
    }

    #[inline(always)]
    fn load_one(word: u64) -> Self {
        // SAFETY: Broadcast/zero intrinsic produces a valid vector register.
        Self(unsafe { _mm_set1_epi64x(word as i64) })
    }

    #[inline(always)]
    fn plus(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm_add_epi64(self.0, rhs.0) })
    }

    #[inline(always)]
    fn plus32(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm_add_epi32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn times(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm_mul_epu32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm_xor_si128(self.0, rhs.0) })
    }

    #[inline(always)]
    fn shl<const BITS: u32>(self) -> Self {
        match BITS {
            // SAFETY: Vector shift intrinsic; shift amount is a compile-time constant.
            1 => Self(unsafe { _mm_slli_epi64(self.0, 1) }),
            // SAFETY: Vector shift intrinsic; shift amount is a compile-time constant.
            2 => Self(unsafe { _mm_slli_epi64(self.0, 2) }),
            // SAFETY: Vector shift intrinsic; shift amount is a compile-time constant.
            3 => Self(unsafe { _mm_slli_epi64(self.0, 3) }),
            // SAFETY: Vector shift intrinsic; shift amount is a compile-time constant.
            32 => Self(unsafe { _mm_slli_epi64(self.0, 32) }),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    fn minus(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm_sub_epi64(self.0, rhs.0) })
    }

    #[inline(always)]
    fn right_shift32(self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm_srli_epi64(self.0, 32) })
    }

    #[inline(always)]
    fn sum(self) -> u64 {
        // SAFETY: See function safety invariants; operands satisfy `Block` / pointer contracts.
        unsafe {
            (_mm_extract_epi64(self.0, 0) as u64).wrapping_add(_mm_extract_epi64(self.0, 1) as u64)
        }
    }
}

/// AVX-512 block: eight 64-bit lanes (`V4Avx512`, `b = 8`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Avx512Block(pub __m512i);

impl Block for Avx512Block {
    const LANES: usize = 8;
    const BYTES: usize = 64;

    #[inline(always)]
    fn load(ptr: *const u8) -> Self {
        // SAFETY: `ptr` addresses at least `Self::BYTES` readable bytes (`Block::load` contract).
        Self(unsafe { _mm512_loadu_si512(ptr as *const __m512i) })
    }

    #[inline(always)]
    fn load_one(word: u64) -> Self {
        // SAFETY: Broadcast/zero intrinsic produces a valid vector register.
        Self(unsafe { _mm512_set1_epi64(word as i64) })
    }

    #[inline(always)]
    fn plus(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm512_add_epi64(self.0, rhs.0) })
    }

    #[inline(always)]
    fn plus32(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm512_add_epi32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn times(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm512_mul_epu32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm512_xor_epi32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn shl<const BITS: u32>(self) -> Self {
        match BITS {
            // SAFETY: Vector shift intrinsic; shift amount is a compile-time constant.
            1 => Self(unsafe { _mm512_slli_epi64(self.0, 1) }),
            // SAFETY: Vector shift intrinsic; shift amount is a compile-time constant.
            2 => Self(unsafe { _mm512_slli_epi64(self.0, 2) }),
            // SAFETY: Vector shift intrinsic; shift amount is a compile-time constant.
            3 => Self(unsafe { _mm512_slli_epi64(self.0, 3) }),
            // SAFETY: Vector shift intrinsic; shift amount is a compile-time constant.
            32 => Self(unsafe { _mm512_slli_epi64(self.0, 32) }),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    fn minus(self, rhs: Self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm512_sub_epi64(self.0, rhs.0) })
    }

    #[inline(always)]
    fn right_shift32(self) -> Self {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        Self(unsafe { _mm512_srli_epi64(self.0, 32) })
    }

    #[inline(always)]
    fn sum(self) -> u64 {
        // SAFETY: SIMD intrinsic operates on valid vector register values.
        unsafe { _mm512_reduce_add_epi64(self.0) as u64 }
    }
}

/// AVX2×2 repeat block (`V4Avx2`, `b = 8`) — reference `RepeatWrapper<BlockWrapper256, 2>`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Avx2Repeat2Block(pub __m256i, pub __m256i);

impl Block for Avx2Repeat2Block {
    const LANES: usize = 8;
    const BYTES: usize = 64;

    #[inline(always)]
    fn zero() -> Self {
        Self(
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_setzero_si256() },
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_setzero_si256() },
        )
    }

    #[inline(always)]
    fn load(ptr: *const u8) -> Self {
        Self(
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_loadu_si256(ptr as *const __m256i) },
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_loadu_si256(ptr.add(32) as *const __m256i) },
        )
    }

    #[inline(always)]
    fn load_one(word: u64) -> Self {
        // SAFETY: Broadcast/zero intrinsic produces a valid vector register.
        let v = unsafe { _mm256_set1_epi64x(word as i64) };
        Self(v, v)
    }

    #[inline(always)]
    fn plus(self, rhs: Self) -> Self {
        Self(
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_add_epi64(self.0, rhs.0) },
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_add_epi64(self.1, rhs.1) },
        )
    }

    #[inline(always)]
    fn plus32(self, rhs: Self) -> Self {
        Self(
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_add_epi32(self.0, rhs.0) },
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_add_epi32(self.1, rhs.1) },
        )
    }

    #[inline(always)]
    fn times(self, rhs: Self) -> Self {
        Self(
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_mul_epu32(self.0, rhs.0) },
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_mul_epu32(self.1, rhs.1) },
        )
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        Self(
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_xor_si256(self.0, rhs.0) },
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_xor_si256(self.1, rhs.1) },
        )
    }

    #[inline(always)]
    fn shl<const BITS: u32>(self) -> Self {
        match BITS {
            1 => Self(
                // SAFETY: SIMD intrinsic operates on valid vector register values.
                unsafe { _mm256_slli_epi64(self.0, 1) },
                // SAFETY: SIMD intrinsic operates on valid vector register values.
                unsafe { _mm256_slli_epi64(self.1, 1) },
            ),
            2 => Self(
                // SAFETY: SIMD intrinsic operates on valid vector register values.
                unsafe { _mm256_slli_epi64(self.0, 2) },
                // SAFETY: SIMD intrinsic operates on valid vector register values.
                unsafe { _mm256_slli_epi64(self.1, 2) },
            ),
            3 => Self(
                // SAFETY: SIMD intrinsic operates on valid vector register values.
                unsafe { _mm256_slli_epi64(self.0, 3) },
                // SAFETY: SIMD intrinsic operates on valid vector register values.
                unsafe { _mm256_slli_epi64(self.1, 3) },
            ),
            32 => Self(
                // SAFETY: SIMD intrinsic operates on valid vector register values.
                unsafe { _mm256_slli_epi64(self.0, 32) },
                // SAFETY: SIMD intrinsic operates on valid vector register values.
                unsafe { _mm256_slli_epi64(self.1, 32) },
            ),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    fn minus(self, rhs: Self) -> Self {
        Self(
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_sub_epi64(self.0, rhs.0) },
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_sub_epi64(self.1, rhs.1) },
        )
    }

    #[inline(always)]
    fn right_shift32(self) -> Self {
        Self(
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_srli_epi64(self.0, 32) },
            // SAFETY: SIMD intrinsic operates on valid vector register values.
            unsafe { _mm256_srli_epi64(self.1, 32) },
        )
    }

    #[inline(always)]
    fn sum(self) -> u64 {
        // SAFETY: See function safety invariants; operands satisfy `Block` / pointer contracts.
        unsafe {
            use core::arch::x86_64::*;
            let a0 = _mm256_extracti128_si256(self.0, 0);
            let a1 = _mm256_extracti128_si256(self.0, 1);
            let b0 = _mm256_extracti128_si256(self.1, 0);
            let b1 = _mm256_extracti128_si256(self.1, 1);
            let ab = _mm_add_epi64(_mm_add_epi64(a0, a1), _mm_add_epi64(b0, b1));
            (_mm_extract_epi64(ab, 0) as u64).wrapping_add(_mm_extract_epi64(ab, 1) as u64)
        }
    }
}
