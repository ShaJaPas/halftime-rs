//! Master key types shared by all [`HalftimeHash`](crate::hash) variants.
//!
//! The number in the name is the **key size in bytes** (`Key32` = 32 bytes), not the tag width.
//! Use [`Tag16`](crate::hash::Tag16) … [`Tag40`](crate::hash::Tag40) for output tags.

use crate::nh;

/// 32-byte master key (default; used by [`KeyInit`](crate::universal_hash::KeyInit)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key32(pub [u8; 32]);

/// 64-byte master key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key64(pub [u8; 64]);

/// 128-byte master key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key128(pub [u8; 128]);

impl From<[u8; 32]> for Key32 {
    fn from(value: [u8; 32]) -> Self {
        Self(value)
    }
}

impl From<[u8; 64]> for Key64 {
    fn from(value: [u8; 64]) -> Self {
        Self(value)
    }
}

impl From<[u8; 128]> for Key128 {
    fn from(value: [u8; 128]) -> Self {
        Self(value)
    }
}

impl AsRef<[u8; 32]> for Key32 {
    fn as_ref(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Default master key type ([`Key32`]).
pub type Key = Key32;

/// Fold any supported master key into 32-byte material for NH entropy expansion.
pub(crate) fn to_material(key: &impl MasterKey) -> [u8; 32] {
    key.to_material()
}

/// Master key material usable with every HalftimeHash variant.
pub trait MasterKey {
    /// 32-byte seed material for NH entropy expansion.
    fn to_material(&self) -> [u8; 32];
}

impl MasterKey for Key32 {
    #[inline]
    fn to_material(&self) -> [u8; 32] {
        self.0
    }
}

impl MasterKey for Key64 {
    fn to_material(&self) -> [u8; 32] {
        fold_key_words(&self.0)
    }
}

impl MasterKey for Key128 {
    fn to_material(&self) -> [u8; 32] {
        fold_key_words(&self.0)
    }
}

impl MasterKey for [u8; 32] {
    #[inline]
    fn to_material(&self) -> [u8; 32] {
        *self
    }
}

/// NH-based folding for longer master keys (independent of message entropy expansion).
fn fold_key_words(key: &[u8]) -> [u8; 32] {
    debug_assert!(key.len() >= 32 && key.len().is_multiple_of(8));
    let words: alloc::vec::Vec<u64> = key
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let seeds = [words[0], words[1], words[2], words[3]];
    let mut out = [0u64; 4];
    for (i, word) in words.iter().enumerate().skip(4) {
        let lane = i & 3;
        out[lane] = nh::hash_word_into(out[lane], *word, seeds[(i + 1) & 3]);
    }
    let mut material = [0u8; 32];
    for (i, word) in out.iter().enumerate() {
        material[i * 8..(i + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
    material
}
