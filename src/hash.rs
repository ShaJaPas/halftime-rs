//! HalftimeHash public types and [`UniversalHash`] implementations.

use core::fmt::{self, Debug};

use alloc::sync::Arc;

use universal_hash::{
    KeyInit, UhfClosure, UniversalHash,
    common::{BlockSizeUser, KeySizeUser},
    consts::U32,
};

use crate::backend::{Absorb, State16, State24, State32, State40};
use crate::entropy::{self, ENTROPY_BLOCK_LANES, EntropyTooShort, prepare_external};
use crate::key::{self, MasterKey};
use crate::variant::{HalftimeVariant, hh16, hh24, hh32, hh40};

/// Size of a HalftimeHash master key in bytes (RustCrypto [`KeyInit`] path).
pub const KEY_SIZE: usize = 32;

macro_rules! define_halftime {
    (
        $(#[$meta:meta])*
        $name:ident,
        $state:ty,
        $block:ty,
        $variant:ty,
        $tag_bytes:expr
    ) => {
        $(#[$meta])*
        pub struct $name {
            state: $state,
        }

        impl KeySizeUser for $name {
            type KeySize = U32;
        }

        impl BlockSizeUser for $name {
            type BlockSize = $block;
        }

        impl KeyInit for $name {
            /// 32-byte key; entropy is expanded on demand via NH (see [`Self::from_master_key`]).
            fn new(key: &universal_hash::Key<Self>) -> Self {
                let mut material = [0u8; KEY_SIZE];
                material.copy_from_slice(key.as_slice());
                Self {
                    state: <$state>::new(&material),
                }
            }
        }

        impl UniversalHash for $name {
            fn update_with_backend(
                &mut self,
                f: impl UhfClosure<BlockSize = Self::BlockSize>,
            ) {
                self.state.update_with_backend(f);
            }

            #[inline]
            fn update_padded(&mut self, data: &[u8]) {
                self.state.absorb(data);
            }

            fn finalize(self) -> universal_hash::Block<Self> {
                self.state.finalize()
            }
        }

        impl $name {
            /// Hash `data` in one shot with a 32-byte RustCrypto key (NH KDF entropy).
            #[must_use]
            pub fn digest(
                key: &universal_hash::Key<Self>,
                data: &[u8],
            ) -> universal_hash::Block<Self> {
                let mut h = Self::new(key);
                h.update_padded(data);
                h.finalize()
            }

            /// Construct from any supported master key ([`Key32`], [`Key64`], [`Key128`]).
            ///
            /// A 32-byte key is expanded into input entropy with NH on demand (convenient KDF).
            /// Longer keys are folded to 32 bytes with NH before expansion.
            #[must_use]
            pub fn from_master_key(key: &impl MasterKey) -> Self {
                let material = key::to_material(key);
                Self {
                    state: <$state>::new(&material),
                }
            }

            /// One-shot hash with a master key of any supported length.
            #[must_use]
            pub fn digest_master_key(
                key: &impl MasterKey,
                data: &[u8],
            ) -> universal_hash::Block<Self> {
                let mut h = Self::from_master_key(key);
                h.update_padded(data);
                h.finalize()
            }

            /// Construct with caller-supplied input entropy (C++ reference style).
            ///
            /// The buffer must hold at least [`Self::entropy_words_needed`] words for the
            /// maximum message length you will hash (defaults to 1 MiB).
            pub fn with_entropy(entropy: &[u64]) -> Result<Self, EntropyTooShort> {
                Self::with_entropy_for(entropy, entropy::DEFAULT_ENTROPY_INPUT)
            }

            /// Like [`Self::with_entropy`], but validate for a specific maximum message size.
            pub fn with_entropy_for(
                entropy: &[u64],
                max_input_bytes: usize,
            ) -> Result<Self, EntropyTooShort> {
                let prepared = prepare_external(
                    Arc::from(entropy),
                    max_input_bytes,
                    ENTROPY_BLOCK_LANES,
                    <$variant as HalftimeVariant>::DIM,
                    <$variant as HalftimeVariant>::ENC,
                    <$variant as HalftimeVariant>::OUT,
                )?;
                Ok(Self {
                    state: <$state>::from_entropy(prepared),
                })
            }

            /// One-shot hash with external entropy sized for `data.len()`.
            pub fn digest_with_entropy(
                entropy: &[u64],
                data: &[u8],
            ) -> Result<universal_hash::Block<Self>, EntropyTooShort> {
                let mut h = Self::with_entropy_for(entropy, data.len())?;
                h.update_padded(data);
                Ok(h.finalize())
            }

            /// Number of `u64` entropy words required to hash up to `input_bytes` of message data.
            #[must_use]
            pub fn entropy_words_needed(input_bytes: usize) -> usize {
                entropy::entropy_words_needed(
                    input_bytes,
                    ENTROPY_BLOCK_LANES,
                    <$variant as HalftimeVariant>::DIM,
                    <$variant as HalftimeVariant>::ENC,
                    <$variant as HalftimeVariant>::OUT,
                )
            }

            /// Tag width in bytes.
            pub const TAG_BYTES: usize = $tag_bytes;
        }

        impl Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name)).finish_non_exhaustive()
            }
        }
    };
}

define_halftime! {
    /// Almost-universal hash with a 16-byte tag (`HalftimeHash16`, Section 4).
    HalftimeHash16,
    State16,
    universal_hash::consts::U16,
    crate::variant::Hh16,
    hh16::TAG_BYTES
}

define_halftime! {
    /// Almost-universal hash with a 24-byte tag (`HalftimeHash24`, Section 4).
    HalftimeHash24,
    State24,
    universal_hash::consts::U24,
    crate::variant::Hh24,
    hh24::TAG_BYTES
}

define_halftime! {
    /// Almost-universal hash with a 32-byte **tag** (`HalftimeHash32`, Section 4).
    ///
    /// Not to be confused with [`Key32`] (32-byte master key).
    HalftimeHash32,
    State32,
    U32,
    crate::variant::Hh32,
    hh32::TAG_BYTES
}

define_halftime! {
    /// Almost-universal hash with a 40-byte tag (`HalftimeHash40`, Section 4).
    HalftimeHash40,
    State40,
    universal_hash::consts::U40,
    crate::variant::Hh40,
    hh40::TAG_BYTES
}

/// Default tag type alias (`HalftimeHash24`).
pub type Tag = Tag24;

/// Tag type for [`HalftimeHash16`].
pub type Tag16 = universal_hash::Block<HalftimeHash16>;
/// Tag type for [`HalftimeHash24`].
pub type Tag24 = universal_hash::Block<HalftimeHash24>;
/// Tag type for [`HalftimeHash32`].
pub type Tag32 = universal_hash::Block<HalftimeHash32>;
/// Tag type for [`HalftimeHash40`].
pub type Tag40 = universal_hash::Block<HalftimeHash40>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::{Key32, Key64, Key128};

    macro_rules! variant_tests {
        ($name:ident, $det:ident, $diff:ident, $len:ident) => {
            #[test]
            fn $det() {
                let key = Key32::from([0xA5u8; 32]);
                let a = $name::digest_master_key(&key, b"hello halftime");
                let b = $name::digest_master_key(&key, b"hello halftime");
                assert_eq!(a, b);
            }

            #[test]
            fn $diff() {
                let key = Key32::from([0xA5u8; 32]);
                let a = $name::digest_master_key(&key, b"alpha");
                let b = $name::digest_master_key(&key, b"beta");
                assert_ne!(a, b);
            }

            #[test]
            fn $len() {
                let key = Key32::from([0xA5u8; 32]);
                let short = $name::digest_master_key(&key, b"x");
                let long = $name::digest_master_key(&key, &[b'x'; 1024]);
                assert_ne!(short, long);
            }
        };
    }

    variant_tests!(HalftimeHash16, hh16_det, hh16_diff, hh16_len);
    variant_tests!(HalftimeHash24, hh24_det, hh24_diff, hh24_len);
    variant_tests!(HalftimeHash32, hh32_det, hh32_diff, hh32_len);
    variant_tests!(HalftimeHash40, hh40_det, hh40_diff, hh40_len);

    #[test]
    fn variants_differ() {
        let key = Key32::from([0xA5u8; 32]);
        let t16 = HalftimeHash16::digest_master_key(&key, b"same");
        let t24 = HalftimeHash24::digest_master_key(&key, b"same");
        assert_ne!(t16.as_slice(), &t24.as_slice()[..16]);
    }

    #[test]
    fn key64_and_key128_work_with_any_variant() {
        let k64 = Key64::from([0x11u8; 64]);
        let k128 = Key128::from([0x22u8; 128]);
        let _ = HalftimeHash16::digest_master_key(&k64, b"a");
        let _ = HalftimeHash40::digest_master_key(&k128, b"b");
    }

    #[test]
    fn key_init_matches_master_key32() {
        let bytes = [0x42u8; 32];
        let ukey = universal_hash::Key::<HalftimeHash24>::from(bytes);
        let mkey = Key32::from(bytes);
        let a = HalftimeHash24::digest(&ukey, b"msg");
        let b = HalftimeHash24::digest_master_key(&mkey, b"msg");
        assert_eq!(a, b);
    }

    #[test]
    fn external_entropy_matches_key_derived() {
        let key = Key32::from([0xA5u8; 32]);
        let key_tag = HalftimeHash24::digest_master_key(&key, b"entropy test");
        let need = HalftimeHash24::entropy_words_needed(b"entropy test".len());
        let mut words = alloc::vec::Vec::with_capacity(need);
        {
            let mut e = entropy::Entropy::from_key_material(&key.0);
            e.ensure_for_input(b"entropy test".len(), ENTROPY_BLOCK_LANES, 7, 9, 3)
                .unwrap();
            words.extend_from_slice(e.as_slice());
        }
        let ext_tag = HalftimeHash24::digest_with_entropy(&words, b"entropy test").unwrap();
        assert_eq!(key_tag, ext_tag);
    }
}
