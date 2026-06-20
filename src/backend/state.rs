//! Generic [`UniversalHash`] state for any [`HalftimeVariant`] and hasher backend.

use core::marker::PhantomData;

use universal_hash::{
    UhfBackend, UhfClosure, UniversalHash,
    array::Array,
    common::{BlockSizeUser, ParBlocksSizeUser},
    consts::U1,
};

use crate::backend::feed::HasherUpdate;
use crate::backend::{Absorb, feed};
use crate::entropy::Entropy;
use crate::variant::HalftimeVariant;

/// Construct a hasher from a 32-byte master key.
pub(crate) trait HasherFromKey: HasherUpdate + Clone + Sized {
    fn from_key(key: &[u8; 32]) -> Self;
}

/// Construct a hasher from a prepared entropy stream.
pub(crate) trait HasherFromEntropy: HasherUpdate + Clone + Sized {
    fn from_entropy(entropy: Entropy) -> Self;
}

/// Route whole-buffer ingest through plain or `target_feature`-aware feed helpers.
pub(crate) trait FeedDispatch {
    fn feed<H: HasherUpdate>(hasher: &mut H, data: &[u8]);
    fn finish<H: HasherUpdate>(hasher: H) -> H::Output;
}

#[derive(Copy, Clone)]
#[cfg(not(all(target_arch = "aarch64", not(halftime_backend = "soft"))))]
pub(crate) struct PlainFeed;

#[cfg(not(all(target_arch = "aarch64", not(halftime_backend = "soft"))))]
impl FeedDispatch for PlainFeed {
    #[inline]
    fn feed<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
        feed::feed(hasher, data);
    }

    #[inline]
    fn finish<H: HasherUpdate>(hasher: H) -> H::Output {
        feed::finish(hasher)
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
#[derive(Copy, Clone)]
pub(crate) struct Avx2Feed;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
impl FeedDispatch for Avx2Feed {
    #[inline]
    fn feed<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
        feed::feed_avx2(hasher, data);
    }

    #[inline]
    fn finish<H: HasherUpdate>(hasher: H) -> H::Output {
        feed::finish_avx2(hasher)
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
#[derive(Copy, Clone)]
pub(crate) struct Sse2Feed;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
impl FeedDispatch for Sse2Feed {
    #[inline]
    fn feed<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
        feed::feed_sse2(hasher, data);
    }

    #[inline]
    fn finish<H: HasherUpdate>(hasher: H) -> H::Output {
        feed::finish_sse2(hasher)
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
#[derive(Copy, Clone)]
pub(crate) struct Avx512Feed;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
impl FeedDispatch for Avx512Feed {
    #[inline]
    fn feed<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
        feed::feed_avx512(hasher, data);
    }

    #[inline]
    fn finish<H: HasherUpdate>(hasher: H) -> H::Output {
        feed::finish_avx512(hasher)
    }
}

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
#[derive(Copy, Clone)]
pub(crate) struct NeonFeed;

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
impl FeedDispatch for NeonFeed {
    #[inline]
    fn feed<H: HasherUpdate>(hasher: &mut H, data: &[u8]) {
        feed::feed_neon(hasher, data);
    }

    #[inline]
    fn finish<H: HasherUpdate>(hasher: H) -> H::Output {
        feed::finish_neon(hasher)
    }
}

#[derive(Clone)]
pub(crate) struct State<V, H, F> {
    hasher: H,
    _marker: PhantomData<(V, F)>,
}

impl<V, H, F> State<V, H, F>
where
    V: HalftimeVariant,
    H: HasherFromKey + HasherFromEntropy,
    F: FeedDispatch,
{
    pub(crate) fn new(key: &[u8; 32]) -> Self {
        Self {
            hasher: H::from_key(key),
            _marker: PhantomData,
        }
    }

    pub(crate) fn from_entropy(entropy: Entropy) -> Self {
        Self {
            hasher: H::from_entropy(entropy),
            _marker: PhantomData,
        }
    }
}

impl<V, H, F> Absorb for State<V, H, F>
where
    V: HalftimeVariant,
    H: HasherUpdate,
    F: FeedDispatch,
{
    #[inline]
    fn absorb(&mut self, data: &[u8]) {
        F::feed(&mut self.hasher, data);
    }
}

impl<V, H, F> BlockSizeUser for State<V, H, F>
where
    V: HalftimeVariant,
{
    type BlockSize = V::TagSize;
}

impl<V, H, F> ParBlocksSizeUser for State<V, H, F>
where
    V: HalftimeVariant,
{
    type ParBlocksSize = U1;
}

impl<V, H, F> UhfBackend for State<V, H, F>
where
    V: HalftimeVariant,
    H: HasherUpdate,
    F: FeedDispatch,
{
    fn proc_block(&mut self, block: &Array<u8, Self::BlockSize>) {
        self.absorb(block.as_slice());
    }
}

impl<V, H, F> UniversalHash for State<V, H, F>
where
    V: HalftimeVariant,
    H: HasherUpdate,
    H::Output: AsRef<[u64]>,
    F: FeedDispatch,
{
    fn update_with_backend(&mut self, f: impl UhfClosure<BlockSize = Self::BlockSize>) {
        f.call(self);
    }

    fn finalize(self) -> Array<u8, Self::BlockSize> {
        let words = F::finish(self.hasher);
        let mut tag = Array::<u8, V::TagSize>::default();
        pack_tag_words(words.as_ref(), tag.as_mut());
        tag
    }
}

fn pack_tag_words(words: &[u64], tag: &mut [u8]) {
    for (i, word) in words.iter().enumerate() {
        tag[i * 8..(i + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
}
