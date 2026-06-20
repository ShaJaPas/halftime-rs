//! NH almost-Δ-universal hash (Section 3.2, UMAC / Badger).

use crate::word::Word;

/// Add two words on 32-bit lanes without cross-lane carry.
#[inline(always)]
pub(crate) fn plus32(a: Word, b: Word) -> Word {
    let lo = (a as u32).wrapping_add(b as u32) as u64;
    let hi = (a >> 32).wrapping_add(b >> 32) << 32;
    lo | hi
}

/// Multiply the low 32-bit halves of two words.
#[inline(always)]
pub(crate) fn mul32(a: Word, b: Word) -> Word {
    let mask = (1u64 << 32) - 1;
    (a & mask).wrapping_mul(b & mask)
}

/// Hash a single word with one entropy word (EHC hash step, step 2).
#[inline(always)]
pub(crate) fn hash_word(data: Word, seed: Word) -> Word {
    let summed = plus32(data, seed);
    mul32(summed, summed >> 32)
}

/// Accumulate one word into a running NH sum (EHC hash step, step 2).
#[inline(always)]
pub(crate) fn hash_word_into(acc: Word, data: Word, seed: Word) -> Word {
    let summed = plus32(data, seed);
    acc.wrapping_add(mul32(summed, summed >> 32))
}

/// Hash a strided row of `width` words with independent seeds.
#[inline(always)]
pub(crate) fn hash_row(data: &[Word], seeds: &[Word]) -> Word {
    let mut acc = hash_word(data[0], seeds[0]);
    for (d, s) in data.iter().zip(seeds.iter()).skip(1) {
        acc = hash_word_into(acc, *d, *s);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline(always)]
    fn pair(d0: u32, d1: u32, s0: u32, s1: u32) -> Word {
        let a = (d0 as u64) + (s0 as u64);
        let b = (d1 as u64) + (s1 as u64);
        a.wrapping_mul(b)
    }

    #[inline(always)]
    fn tree_node_fast(children: &[Word], seeds: &[Word]) -> Word {
        debug_assert_eq!(children.len(), 8);
        debug_assert!(seeds.len() >= 6);
        let mut acc = pair(
            children[0] as u32,
            children[1] as u32,
            seeds[0] as u32,
            seeds[1] as u32,
        );
        acc = acc.wrapping_add(pair(
            children[2] as u32,
            children[3] as u32,
            seeds[2] as u32,
            seeds[3] as u32,
        ));
        acc = acc.wrapping_add(pair(
            children[4] as u32,
            children[5] as u32,
            seeds[4] as u32,
            seeds[5] as u32,
        ));
        acc.wrapping_add(children[6])
            .wrapping_add(children[7].wrapping_shl(32))
    }

    #[test]
    fn tree_node_fast_last_pair_is_additive() {
        let seeds = [0u64; 6];
        let zero_pairs = [0u64, 0, 0, 0, 0, 0, 7, 8];
        assert_eq!(
            tree_node_fast(&zero_pairs, &seeds),
            7u64.wrapping_add(8 << 32)
        );

        let mut only_tail = [0u64; 8];
        only_tail[7] = 1;
        assert_eq!(tree_node_fast(&only_tail, &seeds), 1u64 << 32);
    }
}
