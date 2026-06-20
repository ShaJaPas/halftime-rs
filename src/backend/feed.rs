//! Whole-buffer ingest with optional `#[target_feature]` (poly1305-style SIMD dispatch).

#[cfg(any(
    halftime_backend = "soft",
    all(not(halftime_backend = "soft"), not(target_arch = "aarch64")),
))]
use crate::ehc_badger::Hasher;

pub(crate) trait HasherUpdate {
    fn feed_update(&mut self, data: &[u8]);
    fn feed_finish(self) -> Self::Output;
    type Output;
}

#[cfg(any(
    halftime_backend = "soft",
    all(not(halftime_backend = "soft"), not(target_arch = "aarch64")),
))]
impl<B: crate::block::Block, const DIM: usize, const ENC: usize, const OUT: usize> HasherUpdate
    for Hasher<B, DIM, ENC, OUT>
{
    type Output = [u64; OUT];

    #[inline]
    fn feed_update(&mut self, data: &[u8]) {
        self.update(data);
    }

    #[inline]
    fn feed_finish(self) -> Self::Output {
        self.finalize()
    }
}

macro_rules! impl_fused_hasher_update {
    ($backend:ident) => {
        impl HasherUpdate for crate::ehc_badger::fast::$backend::Hh16 {
            type Output = [u64; 2];

            #[inline]
            fn feed_update(&mut self, data: &[u8]) {
                self.update(data);
            }

            #[inline]
            fn feed_finish(self) -> Self::Output {
                self.finalize()
            }
        }

        impl HasherUpdate for crate::ehc_badger::fast::$backend::Hh24 {
            type Output = [u64; 3];

            #[inline]
            fn feed_update(&mut self, data: &[u8]) {
                self.update(data);
            }

            #[inline]
            fn feed_finish(self) -> Self::Output {
                self.finalize()
            }
        }

        impl HasherUpdate for crate::ehc_badger::fast::$backend::Hh32 {
            type Output = [u64; 4];

            #[inline]
            fn feed_update(&mut self, data: &[u8]) {
                self.update(data);
            }

            #[inline]
            fn feed_finish(self) -> Self::Output {
                self.finalize()
            }
        }

        impl HasherUpdate for crate::ehc_badger::fast::$backend::Hh40 {
            type Output = [u64; 5];

            #[inline]
            fn feed_update(&mut self, data: &[u8]) {
                self.update(data);
            }

            #[inline]
            fn feed_finish(self) -> Self::Output {
                self.finalize()
            }
        }
    };
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
impl_fused_hasher_update!(avx512);

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
impl_fused_hasher_update!(avx2);

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
impl_fused_hasher_update!(sse2);

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
impl_fused_hasher_update!(neon);

#[inline]
pub(crate) fn feed<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
    hasher.feed_update(data);
}

#[inline]
pub(crate) fn finish<H: HasherUpdate>(hasher: H) -> H::Output {
    hasher.feed_finish()
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft"),
    target_feature = "avx2"
))]
#[inline]
pub(crate) fn feed_avx2<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
    feed(hasher, data);
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft"),
    not(target_feature = "avx2")
))]
#[inline]
pub(crate) fn feed_avx2<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
    // SAFETY: Runtime CPU feature detection selected AVX2 before this `target_feature` entry point.
    unsafe { feed_avx2_impl(hasher, data) }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft"),
    target_feature = "avx2"
))]
#[inline]
pub(crate) fn finish_avx2<H: HasherUpdate>(hasher: H) -> H::Output {
    finish(hasher)
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft"),
    not(target_feature = "avx2")
))]
#[inline]
pub(crate) fn finish_avx2<H: HasherUpdate>(hasher: H) -> H::Output {
    // SAFETY: Runtime CPU feature detection selected AVX2 before this `target_feature` entry point.
    unsafe { finish_avx2_impl(hasher) }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft"),
    not(target_feature = "avx2")
))]
#[target_feature(enable = "avx2")]
unsafe fn feed_avx2_impl<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
    hasher.feed_update(data);
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft"),
    not(target_feature = "avx2")
))]
#[target_feature(enable = "avx2")]
unsafe fn finish_avx2_impl<H: HasherUpdate>(hasher: H) -> H::Output {
    hasher.feed_finish()
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
#[inline]
pub(crate) fn feed_sse2<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
    // SAFETY: Runtime CPU feature detection selected SSE2 before this `target_feature` entry point.
    unsafe { feed_sse2_impl(hasher, data) }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
#[inline]
pub(crate) fn finish_sse2<H: HasherUpdate>(hasher: H) -> H::Output {
    // SAFETY: Runtime CPU feature detection selected SSE2 before this `target_feature` entry point.
    unsafe { finish_sse2_impl(hasher) }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
#[target_feature(enable = "sse2")]
unsafe fn feed_sse2_impl<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
    feed(hasher, data);
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
#[target_feature(enable = "sse2")]
unsafe fn finish_sse2_impl<H: HasherUpdate>(hasher: H) -> H::Output {
    finish(hasher)
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
#[inline]
pub(crate) fn feed_avx512<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
    // SAFETY: Runtime CPU feature detection selected AVX-512 before this `target_feature` entry point.
    unsafe { feed_avx512_impl(hasher, data) }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
#[inline]
pub(crate) fn finish_avx512<H: HasherUpdate>(hasher: H) -> H::Output {
    // SAFETY: Runtime CPU feature detection selected AVX-512 before this `target_feature` entry point.
    unsafe { finish_avx512_impl(hasher) }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
#[target_feature(enable = "avx512f")]
unsafe fn feed_avx512_impl<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
    feed(hasher, data);
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
#[target_feature(enable = "avx512f")]
unsafe fn finish_avx512_impl<H: HasherUpdate>(hasher: H) -> H::Output {
    finish(hasher)
}

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
#[inline]
pub(crate) fn feed_neon<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
    // SAFETY: Runtime CPU feature detection selected NEON before this `target_feature` entry point.
    unsafe { feed_neon_impl(hasher, data) }
}

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
#[inline]
pub(crate) fn finish_neon<H: HasherUpdate>(hasher: H) -> H::Output {
    // SAFETY: Runtime CPU feature detection selected NEON before this `target_feature` entry point.
    unsafe { finish_neon_impl(hasher) }
}

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
#[target_feature(enable = "neon")]
unsafe fn feed_neon_impl<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
    feed(hasher, data);
}

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
#[target_feature(enable = "neon")]
unsafe fn finish_neon_impl<H: HasherUpdate>(hasher: H) -> H::Output {
    finish(hasher)
}
