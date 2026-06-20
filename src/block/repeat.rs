//! Repeat block wrapper (reference `RepeatWrapper`, widens `b` on any inner block).

use super::Block;

/// `count` inner blocks loaded/stored contiguously (`V2`/`V3`/`V4` scalar repeats).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RepeatBlock<B: Block, const COUNT: usize>(pub [B; COUNT]);

impl<B: Block, const COUNT: usize> Block for RepeatBlock<B, COUNT> {
    const LANES: usize = B::LANES * COUNT;
    const BYTES: usize = B::BYTES * COUNT;

    #[inline(always)]
    fn load(ptr: *const u8) -> Self {
        let mut inner = [B::zero(); COUNT];
        for (i, slot) in inner.iter_mut().enumerate() {
            // SAFETY: Loop index `i < COUNT` keeps the load pointer inside this `RepeatBlock`'s span.
            *slot = B::load(unsafe { ptr.add(i * B::BYTES) });
        }
        Self(inner)
    }

    #[inline(always)]
    fn load_one(word: u64) -> Self {
        Self([B::load_one(word); COUNT])
    }

    #[inline(always)]
    fn plus(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].plus(rhs.0[i])))
    }

    #[inline(always)]
    fn plus32(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].plus32(rhs.0[i])))
    }

    #[inline(always)]
    fn times(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].times(rhs.0[i])))
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].xor(rhs.0[i])))
    }

    #[inline(always)]
    fn shl<const BITS: u32>(self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].shl::<BITS>()))
    }

    #[inline(always)]
    fn minus(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].minus(rhs.0[i])))
    }

    #[inline(always)]
    fn right_shift32(self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].right_shift32()))
    }

    #[inline(always)]
    fn sum(self) -> u64 {
        let mut acc = 0u64;
        for block in self.0 {
            acc = acc.wrapping_add(block.sum());
        }
        acc
    }
}
