//! Input-entropy stream for HalftimeHash (Section 4.3).
//!
//! Two modes:
//! - **Key-derived** (default): a 32-byte master key is expanded on demand via NH (`prf_word`).
//! - **External** (C++-style): caller supplies a pre-generated `&[u64]` buffer.

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::nh;

const FANOUT: usize = 8;
const IN_W: usize = 3;
const MAX_STACK: usize = 9;

/// SIMD macro-block width used for [`entropy_words_needed`] planning (8 on x86 AVX-512).
pub const ENTROPY_BLOCK_LANES: usize = 8;

/// External entropy buffer is shorter than required for the message length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntropyTooShort {
    /// Required number of `u64` words.
    pub need_words: usize,
    /// Provided number of `u64` words.
    pub got_words: usize,
}

/// Expands a master key or wraps caller-supplied input entropy.
#[derive(Clone)]
pub(crate) struct Entropy {
    prf_seeds: Option<[u64; 4]>,
    words: Arc<[u64]>,
}

impl Entropy {
    /// Key-derived entropy: NH expands 32-byte material into a word stream.
    pub(crate) fn from_key_material(key: &[u8; 32]) -> Self {
        Self {
            prf_seeds: Some([
                u64::from_le_bytes(key[24..32].try_into().unwrap()),
                u64::from_le_bytes(key[..8].try_into().unwrap()),
                u64::from_le_bytes(key[8..16].try_into().unwrap()),
                u64::from_le_bytes(key[16..24].try_into().unwrap()),
            ]),
            words: Arc::from([]),
        }
    }

    /// External entropy (paper / C++ reference style): use `words` as-is.
    pub(crate) fn from_words(words: Arc<[u64]>) -> Self {
        Self {
            prf_seeds: None,
            words,
        }
    }

    /// Pre-expand or validate entropy for hashing an input of `input_bytes`.
    pub(crate) fn ensure_for_input(
        &mut self,
        input_bytes: usize,
        block_lanes: usize,
        dim: usize,
        enc: usize,
        out: usize,
    ) -> Result<(), EntropyTooShort> {
        let need = entropy_words_needed(input_bytes, block_lanes, dim, enc, out);
        if self.prf_seeds.is_none() {
            if self.words.len() >= need {
                return Ok(());
            }
            return Err(EntropyTooShort {
                need_words: need,
                got_words: self.words.len(),
            });
        }
        if self.words.len() >= need {
            return Ok(());
        }
        let mut vec = Vec::with_capacity(need);
        vec.extend_from_slice(&self.words);
        let seeds = self.prf_seeds.expect("key-derived entropy");
        for index in vec.len()..need {
            vec.push(prf_word(index, &seeds));
        }
        self.words = Arc::from(vec);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[u64] {
        &self.words
    }

    /// Grow or validate the word stream for hashing `input_bytes` of message data.
    pub(crate) fn ensure_for_input_or_panic(
        &mut self,
        input_bytes: usize,
        block_lanes: usize,
        dim: usize,
        enc: usize,
        out: usize,
    ) {
        if let Err(err) = self.ensure_for_input(input_bytes, block_lanes, dim, enc, out) {
            panic!(
                "external entropy buffer too short (need {} u64 words, have {}); \
                 call HalftimeHash*::entropy_words_needed(input_len)",
                err.need_words, err.got_words
            );
        }
    }
}

/// Default pre-expansion size for key-derived entropy (1 MiB of message data).
pub(crate) const DEFAULT_ENTROPY_INPUT: usize = 1 << 20;

/// Build key-derived entropy pre-expanded for [`DEFAULT_ENTROPY_INPUT`].
pub(crate) fn entropy_for_key(
    key: &[u8; 32],
    block_lanes: usize,
    dim: usize,
    enc: usize,
    out: usize,
) -> Entropy {
    let mut entropy = Entropy::from_key_material(key);
    entropy.ensure_for_input_or_panic(DEFAULT_ENTROPY_INPUT, block_lanes, dim, enc, out);
    entropy
}

/// Wrap caller-supplied entropy and validate length for `max_input_bytes` of message data.
pub(crate) fn prepare_external(
    words: Arc<[u64]>,
    max_input_bytes: usize,
    block_lanes: usize,
    dim: usize,
    enc: usize,
    out: usize,
) -> Result<Entropy, EntropyTooShort> {
    let mut entropy = Entropy::from_words(words);
    entropy.ensure_for_input(max_input_bytes.max(1), block_lanes, dim, enc, out)?;
    Ok(entropy)
}

#[inline]
fn prf_word(index: usize, seeds: &[u64; 4]) -> u64 {
    let block = [index as u64, seeds[1], seeds[2], seeds[3]];
    nh::hash_row(&block, seeds)
}

/// `FloorLog` from the reference (`halftime-hash.hpp`).
#[inline]
const fn floor_log(a: usize, b: usize) -> usize {
    if b == 0 || b < a {
        0
    } else {
        1 + floor_log(a, b / a)
    }
}

/// Number of 64-bit entropy words for input length `n` (reference `GetEntropyBytesNeeded / 8`).
///
/// Tree entropy is reserved for [`MAX_STACK`] levels (see reference `Hash()`), not only the
/// height `h` implied by `n`.
#[inline]
pub fn entropy_words_needed(
    n: usize,
    block_lanes: usize,
    dim: usize,
    enc: usize,
    out: usize,
) -> usize {
    let b = block_lanes;
    let macro_words = dim * IN_W;
    let h = if macro_words == 0 || b == 0 {
        0
    } else {
        floor_log(FANOUT, n / (b * macro_words))
    };
    let tree_reserved = (FANOUT - 1) * out * MAX_STACK;
    enc * IN_W + tree_reserved + b * FANOUT * out * h + b * dim * IN_W + out - 1
}

/// Base index of tree-hash entropy after the EHC seed block.
#[inline]
pub(crate) const fn tree_entropy_base(encoded_len: usize, stride: usize) -> usize {
    encoded_len * stride
}

/// Entropy index for tree level `level`.
#[inline]
pub(crate) const fn tree_entropy_level(
    encoded_len: usize,
    stride: usize,
    fanout: usize,
    k: usize,
    level: usize,
) -> usize {
    tree_entropy_base(encoded_len, stride) + level * (fanout - 1) * k
}

/// First finalizer entropy index after the tree region (`kMaxStack = 9`).
#[inline]
pub(crate) const fn finalizer_entropy_base(
    encoded_len: usize,
    stride: usize,
    fanout: usize,
    k: usize,
) -> usize {
    encoded_len * stride + k * (fanout - 1) * MAX_STACK
}

#[cfg(test)]
mod tests {
    use super::*;

    const HH24_DATA_ROWS: usize = 7;
    const HH24_ENCODED_LEN: usize = 9;
    const HH24_K: usize = 3;

    #[test]
    fn preexpanded_is_deterministic() {
        let key = [0xA5u8; 32];
        let mut a = Entropy::from_key_material(&key);
        let mut b = Entropy::from_key_material(&key);
        a.ensure_for_input(4096, 8, HH24_DATA_ROWS, HH24_ENCODED_LEN, HH24_K)
            .unwrap();
        b.ensure_for_input(4096, 8, HH24_DATA_ROWS, HH24_ENCODED_LEN, HH24_K)
            .unwrap();
        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn incremental_expand_matches_one_shot() {
        let key = [0xA5u8; 32];
        let mut incremental = Entropy::from_key_material(&key);
        incremental
            .ensure_for_input(512, 8, HH24_DATA_ROWS, HH24_ENCODED_LEN, HH24_K)
            .unwrap();
        incremental
            .ensure_for_input(4096, 8, HH24_DATA_ROWS, HH24_ENCODED_LEN, HH24_K)
            .unwrap();
        let mut one_shot = Entropy::from_key_material(&key);
        one_shot
            .ensure_for_input(4096, 8, HH24_DATA_ROWS, HH24_ENCODED_LEN, HH24_K)
            .unwrap();
        assert_eq!(incremental.as_slice(), one_shot.as_slice());
    }

    #[test]
    fn external_entropy_rejects_short_buffer() {
        let words = Arc::from([1u64, 2, 3]);
        let mut e = Entropy::from_words(words);
        let err = e
            .ensure_for_input(1 << 20, 8, HH24_DATA_ROWS, HH24_ENCODED_LEN, HH24_K)
            .unwrap_err();
        assert!(err.need_words > err.got_words);
    }

    #[test]
    fn clone_shares_words() {
        let key = [0xA5u8; 32];
        let mut a = Entropy::from_key_material(&key);
        a.ensure_for_input(4096, 8, HH24_DATA_ROWS, HH24_ENCODED_LEN, HH24_K)
            .unwrap();
        let b = a.clone();
        assert_eq!(Arc::as_ptr(&a.words), Arc::as_ptr(&b.words));
    }

    #[test]
    fn one_mib_hh24_entropy_size() {
        let words = entropy_words_needed(1024 * 1024, 8, HH24_DATA_ROWS, HH24_ENCODED_LEN, HH24_K);
        assert!(words * 8 >= 9000);
        assert!(words * 8 <= 10000);
    }

    #[test]
    fn short_input_covers_finalizer_base() {
        let words = entropy_words_needed(16, 8, HH24_DATA_ROWS, HH24_ENCODED_LEN, HH24_K);
        assert!(words > finalizer_entropy_base(HH24_ENCODED_LEN, IN_W, FANOUT, HH24_K));
    }
}
