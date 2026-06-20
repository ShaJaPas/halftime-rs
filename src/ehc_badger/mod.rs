//! Unified EHC + tree + greedy finalizer (`EhcBadger` in the reference).

#![cfg_attr(
    all(target_arch = "aarch64", not(halftime_backend = "soft")),
    allow(dead_code)
)]

mod combine;
mod encode;

#[cfg(all(
    not(halftime_backend = "soft"),
    any(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_arch = "aarch64",
    )
))]
pub(crate) mod fast;

use crate::block::{Block, multiply_add};
use crate::entropy::{self, Entropy};

pub(crate) use combine::{combine2, combine3, combine4, combine5};
pub(crate) use encode::{encode2, encode3, encode4, encode5};

const MAX_STACK: usize = 9;
const FANOUT: usize = 8;
const IN_W: usize = 3;

#[inline(always)]
fn mix<B: Block>(accum: B, input: B, seed: B) -> B {
    let output = input.plus32(seed);
    let twin = output.right_shift32();
    multiply_add(accum, output, twin)
}

#[inline(always)]
fn mix_one<B: Block>(accum: B, input: B, seed: u64) -> B {
    mix(accum, input, B::load_one(seed))
}

#[inline(always)]
fn mix_none<B: Block>(input: B, seed: u64) -> B {
    let output = input.plus32(B::load_one(seed));
    let twin = output.right_shift32();
    output.times(twin)
}

#[inline(always)]
fn load_block_seed<B: Block>(words: &[u64], index: usize) -> B {
    debug_assert!(index + B::LANES <= words.len());
    if B::LANES == 1 {
        B::load_one(words[index])
    } else {
        // SAFETY: `words` entropy slice is large enough for the indexed seed block.
        unsafe { B::load(words.as_ptr().add(index) as *const u8) }
    }
}

/// Reference `EhcBadger::Load` from sequential wide blocks (`LoadBlock` per cell).
#[inline(always)]
fn load_strided<B: Block, const DIM: usize>(input: &[u8]) -> [[B; IN_W]; DIM] {
    let mut out = [[B::zero(); IN_W]; DIM];
    for (i, row) in out.iter_mut().enumerate().take(DIM) {
        for (j, cell) in row.iter_mut().enumerate().take(IN_W) {
            let off = (i * IN_W + j) * B::BYTES;
            *cell = B::load(input[off..].as_ptr());
        }
    }
    out
}

/// Gather one wide block from `lanes` instance-major rows (reference batch layout).
#[inline(always)]
fn load_gathered<B: Block>(src: &[u8], lanes: usize, instance_bytes: usize, word_idx: usize) -> B {
    debug_assert!(lanes <= B::LANES);
    if B::LANES == 1 {
        let off = word_idx * 8;
        debug_assert!(off + 8 <= instance_bytes);
        let mut word = 0u64;
        // SAFETY: See function safety invariants; operands satisfy `Block` / pointer contracts.
        unsafe {
            core::ptr::copy_nonoverlapping(
                src.as_ptr().add(off),
                &mut word as *mut u64 as *mut u8,
                8,
            );
        }
        return B::load_one(word);
    }
    let mut bytes = [0u8; 64];
    debug_assert!(B::BYTES <= bytes.len());
    for lane in 0..B::LANES {
        let dst = lane * 8;
        if lane < lanes {
            let src_off = lane * instance_bytes + word_idx * 8;
            bytes[dst..dst + 8].copy_from_slice(&src[src_off..src_off + 8]);
        }
    }
    B::load(bytes.as_ptr())
}

#[inline(always)]
fn load_strided_from_instances<B: Block, const DIM: usize>(
    src: &[u8],
    lanes: usize,
    instance_bytes: usize,
) -> [[B; IN_W]; DIM] {
    let mut out = [[B::zero(); IN_W]; DIM];
    for (i, row) in out.iter_mut().enumerate().take(DIM) {
        for (j, cell) in row.iter_mut().enumerate().take(IN_W) {
            *cell = load_gathered::<B>(src, lanes, instance_bytes, i * IN_W + j);
        }
    }
    out
}

fn encode<B: Block, const ENC: usize, const OUT: usize>(io: &mut [[B; IN_W]; ENC]) {
    match OUT {
        // SAFETY: `ENC` matches Gabrielyan row count for this `OUT` width; buffer layout is `[[B; IN_W]; ENC]`.
        2 => encode2(unsafe { &mut *(io.as_mut_ptr() as *mut [[B; IN_W]; 7]) }),
        // SAFETY: `ENC` matches Gabrielyan row count for this `OUT` width; buffer layout is `[[B; IN_W]; ENC]`.
        3 => encode3(unsafe { &mut *(io.as_mut_ptr() as *mut [[B; IN_W]; 9]) }),
        // SAFETY: `ENC` matches Gabrielyan row count for this `OUT` width; buffer layout is `[[B; IN_W]; ENC]`.
        4 => encode4(unsafe { &mut *(io.as_mut_ptr() as *mut [[B; IN_W]; 10]) }),
        // SAFETY: `ENC` matches Gabrielyan row count for this `OUT` width; buffer layout is `[[B; IN_W]; ENC]`.
        5 => encode5(unsafe { &mut *(io.as_mut_ptr() as *mut [[B; IN_W]; 9]) }),
        _ => unreachable!(),
    }
}

fn combine<B: Block, const ENC: usize, const OUT: usize>(input: &[B; ENC]) -> [B; OUT] {
    match OUT {
        2 => {
            // SAFETY: `ENC` matches combine input row count; buffer layout is `[B; ENC]`.
            let r: [B; 2] = combine2(unsafe { &*(input.as_ptr() as *const [B; 7]) });
            let mut out = [B::zero(); OUT];
            out[..2].copy_from_slice(&r);
            out
        }
        3 => {
            // SAFETY: `ENC` matches combine input row count; buffer layout is `[B; ENC]`.
            let r: [B; 3] = combine3(unsafe { &*(input.as_ptr() as *const [B; 9]) });
            let mut out = [B::zero(); OUT];
            out[..3].copy_from_slice(&r);
            out
        }
        4 => {
            // SAFETY: `ENC` matches combine input row count; buffer layout is `[B; ENC]`.
            let r: [B; 4] = combine4(unsafe { &*(input.as_ptr() as *const [B; 10]) });
            let mut out = [B::zero(); OUT];
            out[..4].copy_from_slice(&r);
            out
        }
        5 => {
            // SAFETY: `ENC` matches combine input row count; buffer layout is `[B; ENC]`.
            let r: [B; 5] = combine5(unsafe { &*(input.as_ptr() as *const [B; 9]) });
            let mut out = [B::zero(); OUT];
            out[..5].copy_from_slice(&r);
            out
        }
        _ => unreachable!(),
    }
}

/// Reference `EhcBadger::Hash` (column `j`, then row `i`).
#[inline(always)]
fn hash_rows<B: Block, const ENC: usize>(
    input: &[[B; IN_W]; ENC],
    words: &[u64],
    output: &mut [B; ENC],
) {
    for i in 0..ENC {
        output[i] = mix_none(input[i][0], words[i * IN_W]);
    }
    for j in 1..IN_W {
        for i in 0..ENC {
            output[i] = mix_one(output[i], input[i][j], words[i * IN_W + j]);
        }
    }
}

#[inline(always)]
fn ehc_base_layer<B: Block, const DIM: usize, const ENC: usize, const OUT: usize>(
    input: &[u8],
    entropy: &Entropy,
    output: &mut [B; OUT],
) {
    let mut scratch = [[B::zero(); IN_W]; ENC];
    let loaded = load_strided::<B, DIM>(input);
    scratch[..DIM].copy_from_slice(&loaded);
    encode::<B, ENC, OUT>(&mut scratch);
    let mut hashed = [B::zero(); ENC];
    hash_rows::<B, ENC>(&scratch, entropy.as_slice(), &mut hashed);
    *output = combine::<B, ENC, OUT>(&hashed);
}

fn ehc_base_layer_from_instances<B: Block, const DIM: usize, const ENC: usize, const OUT: usize>(
    src: &[u8],
    lanes: usize,
    instance_bytes: usize,
    entropy: &Entropy,
    output: &mut [B; OUT],
) {
    let mut scratch = [[B::zero(); IN_W]; ENC];
    let loaded = load_strided_from_instances::<B, DIM>(src, lanes, instance_bytes);
    scratch[..DIM].copy_from_slice(&loaded);
    encode::<B, ENC, OUT>(&mut scratch);
    let mut hashed = [B::zero(); ENC];
    hash_rows::<B, ENC>(&scratch, entropy.as_slice(), &mut hashed);
    *output = combine::<B, ENC, OUT>(&hashed);
}

/// Reference `EhcBadger::EhcUpperLayer` (`MixOne` chain, not `tree_node_fast`).
#[inline(always)]
fn ehc_upper_layer<B: Block, const ENC: usize, const OUT: usize>(
    children: &[[B; OUT]; FANOUT],
    level: usize,
    entropy: &Entropy,
    output: &mut [B; OUT],
) {
    let words = entropy.as_slice();
    let base = entropy::tree_entropy_level(ENC, IN_W, FANOUT, OUT, level);
    for i in 0..OUT {
        output[i] = children[0][i];
        for j in 1..FANOUT {
            let seed = words[base + (FANOUT - 1) * i + (j - 1)];
            output[i] = mix_one(output[i], children[j][i], seed);
        }
    }
}

struct BlockGreedy<B: Block, const OUT: usize> {
    accum: [B; OUT],
    seed_index: usize,
}

impl<B: Block, const OUT: usize> BlockGreedy<B, OUT> {
    fn new(start: usize) -> Self {
        Self {
            accum: [B::zero(); OUT],
            seed_index: start,
        }
    }

    fn insert_chunk(&mut self, chunk: &[B; OUT], words: &[u64]) {
        for (acc, &word) in self.accum.iter_mut().zip(chunk.iter()) {
            let seed = load_block_seed::<B>(words, self.seed_index);
            *acc = mix(*acc, word, seed);
            self.seed_index += B::LANES;
        }
    }

    fn insert_toeplitz(&mut self, word: B, words: &[u64]) {
        for (i, acc) in self.accum.iter_mut().enumerate() {
            let seed = load_block_seed::<B>(words, self.seed_index + i * B::LANES);
            *acc = mix(*acc, word, seed);
        }
        self.seed_index += B::LANES;
    }

    fn finish(self) -> [u64; OUT] {
        core::array::from_fn(|i| self.accum[i].sum())
    }
}

/// Incremental hasher (strided / SIMD-aware).
#[derive(Clone)]
pub(crate) struct Hasher<B: Block, const DIM: usize, const ENC: usize, const OUT: usize> {
    stack: [[[B; OUT]; FANOUT]; MAX_STACK],
    stack_lengths: [u8; MAX_STACK],
    buffer: [u8; 2048],
    buffer_len: usize,
    total_len: u64,
    entropy: Entropy,
}

impl<B: Block, const DIM: usize, const ENC: usize, const OUT: usize> Hasher<B, DIM, ENC, OUT> {
    pub(crate) const INSTANCE_BYTES: usize = DIM * IN_W * 8;
    const MACRO_BYTES: usize = DIM * IN_W * B::BYTES;

    pub(crate) fn new(key: &[u8; 32]) -> Self {
        Self::from_prepared_entropy(entropy::entropy_for_key(key, B::LANES, DIM, ENC, OUT))
    }

    pub(crate) fn from_prepared_entropy(entropy: Entropy) -> Self {
        Self {
            stack: core::array::from_fn(|_| core::array::from_fn(|_| [B::zero(); OUT])),
            stack_lengths: [0; MAX_STACK],
            buffer: [0; 2048],
            buffer_len: 0,
            total_len: 0,
            entropy,
        }
    }

    fn ensure_entropy(&mut self) {
        let nbytes = (self.total_len as usize).max(1);
        self.entropy
            .ensure_for_input_or_panic(nbytes, B::LANES, DIM, ENC, OUT);
    }

    /// Reference `DfsTreeHash`: one wide macro block per iteration.
    #[inline(always)]
    fn push_wide_block(&mut self, input: &[u8]) {
        debug_assert!(input.len() >= Self::MACRO_BYTES);
        let mut i = 0usize;
        while i < MAX_STACK && self.stack_lengths[i] as usize == FANOUT {
            i += 1;
        }
        for j in (0..i).rev() {
            let mut combined = [B::zero(); OUT];
            ehc_upper_layer::<B, ENC, OUT>(&self.stack[j], j, &self.entropy, &mut combined);
            let slot = self.stack_lengths[j + 1] as usize;
            self.stack[j + 1][slot] = combined;
            self.stack_lengths[j] = 0;
            self.stack_lengths[j + 1] += 1;
        }
        let mut out = [B::zero(); OUT];
        ehc_base_layer::<B, DIM, ENC, OUT>(&input[..Self::MACRO_BYTES], &self.entropy, &mut out);
        let slot = self.stack_lengths[0] as usize;
        self.stack[0][slot] = out;
        self.stack_lengths[0] += 1;
    }

    fn push_instance_batch(&mut self, src: &[u8], lanes: usize) {
        debug_assert!(lanes <= B::LANES);
        debug_assert!(src.len() >= lanes * Self::INSTANCE_BYTES);
        let mut i = 0usize;
        while i < MAX_STACK && self.stack_lengths[i] as usize == FANOUT {
            i += 1;
        }
        for j in (0..i).rev() {
            let mut combined = [B::zero(); OUT];
            ehc_upper_layer::<B, ENC, OUT>(&self.stack[j], j, &self.entropy, &mut combined);
            let slot = self.stack_lengths[j + 1] as usize;
            self.stack[j + 1][slot] = combined;
            self.stack_lengths[j] = 0;
            self.stack_lengths[j + 1] += 1;
        }
        let mut out = [B::zero(); OUT];
        ehc_base_layer_from_instances::<B, DIM, ENC, OUT>(
            src,
            lanes,
            Self::INSTANCE_BYTES,
            &self.entropy,
            &mut out,
        );
        let slot = self.stack_lengths[0] as usize;
        self.stack[0][slot] = out;
        self.stack_lengths[0] += 1;
    }

    pub(crate) fn update(&mut self, data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        self.ensure_entropy();
        let mut off = 0;

        if self.buffer_len > 0 {
            let need = Self::INSTANCE_BYTES - self.buffer_len;
            let take = need.min(data.len());
            self.buffer[self.buffer_len..self.buffer_len + take].copy_from_slice(&data[..take]);
            self.buffer_len += take;
            off += take;
            if self.buffer_len == Self::INSTANCE_BYTES {
                let chunk = self.buffer;
                if B::LANES == 1 {
                    self.push_wide_block(&chunk[..Self::MACRO_BYTES]);
                } else {
                    self.push_instance_batch(&chunk[..Self::INSTANCE_BYTES], 1);
                }
                self.buffer_len = 0;
            }
        }

        while off + Self::MACRO_BYTES <= data.len() {
            let block_count = (data.len() - off) / Self::MACRO_BYTES;
            // SAFETY: Loop guard `off + block_count * MACRO_BYTES <= data.len()` keeps the base in bounds.
            let base = unsafe { data.as_ptr().add(off) };
            for k in 0..block_count {
                // SAFETY: See function safety invariants; operands satisfy `Block` / pointer contracts.
                let chunk = unsafe {
                    core::slice::from_raw_parts(base.add(k * Self::MACRO_BYTES), Self::MACRO_BYTES)
                };
                self.push_wide_block(chunk);
            }
            off += block_count * Self::MACRO_BYTES;
        }

        let tail = &data[off..];
        if !tail.is_empty() {
            self.buffer[..tail.len()].copy_from_slice(tail);
            self.buffer_len = tail.len();
        }
    }

    pub(crate) fn finalize(mut self) -> [u64; OUT] {
        self.ensure_entropy();
        let fin_base = entropy::finalizer_entropy_base(ENC, IN_W, FANOUT, OUT);
        let words = self.entropy.as_slice();
        let mut greedy = BlockGreedy::<B, OUT>::new(fin_base);

        for level in 0..MAX_STACK {
            for slot in 0..self.stack_lengths[level] as usize {
                greedy.insert_chunk(&self.stack[level][slot], words);
            }
        }

        let tail_len = self.buffer_len;
        let mut i = 0;
        while i + B::BYTES <= tail_len {
            let word = B::load(self.buffer[i..].as_ptr());
            greedy.insert_toeplitz(word, words);
            i += B::BYTES;
        }
        {
            let mut extra = [0u8; 64];
            if i < tail_len {
                extra[..tail_len - i].copy_from_slice(&self.buffer[i..tail_len]);
            }
            greedy.insert_toeplitz(B::load(extra.as_ptr()), words);
        }

        greedy.finish()
    }
}

#[cfg(halftime_backend = "soft")]
pub(crate) type Hasher16 = Hasher<crate::block::ScalarBlock, 6, 7, 2>;
#[cfg(halftime_backend = "soft")]
pub(crate) type Hasher24 = Hasher<crate::block::ScalarBlock, 7, 9, 3>;
#[cfg(halftime_backend = "soft")]
pub(crate) type Hasher32 = Hasher<crate::block::ScalarBlock, 7, 10, 4>;
#[cfg(halftime_backend = "soft")]
pub(crate) type Hasher40 = Hasher<crate::block::ScalarBlock, 5, 9, 5>;

#[cfg(all(not(halftime_backend = "soft"), not(target_arch = "aarch64")))]
pub(crate) type Repeat8Hasher16 =
    Hasher<crate::block::RepeatBlock<crate::block::ScalarBlock, 8>, 6, 7, 2>;
#[cfg(all(not(halftime_backend = "soft"), not(target_arch = "aarch64")))]
pub(crate) type Repeat8Hasher24 =
    Hasher<crate::block::RepeatBlock<crate::block::ScalarBlock, 8>, 7, 9, 3>;
#[cfg(all(not(halftime_backend = "soft"), not(target_arch = "aarch64")))]
pub(crate) type Repeat8Hasher32 =
    Hasher<crate::block::RepeatBlock<crate::block::ScalarBlock, 8>, 7, 10, 4>;
#[cfg(all(not(halftime_backend = "soft"), not(target_arch = "aarch64")))]
pub(crate) type Repeat8Hasher40 =
    Hasher<crate::block::RepeatBlock<crate::block::ScalarBlock, 8>, 5, 9, 5>;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx512Hasher16 = fast::avx512::Hh16;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx512Hasher24 = fast::avx512::Hh24;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx512Hasher32 = fast::avx512::Hh32;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx512Hasher40 = fast::avx512::Hh40;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx2Repeat8Hasher16 = fast::avx2::Hh16;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx2Repeat8Hasher24 = fast::avx2::Hh24;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx2Repeat8Hasher32 = fast::avx2::Hh32;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx2Repeat8Hasher40 = fast::avx2::Hh40;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Sse2Repeat8Hasher16 = fast::sse2::Hh16;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Sse2Repeat8Hasher24 = fast::sse2::Hh24;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Sse2Repeat8Hasher32 = fast::sse2::Hh32;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Sse2Repeat8Hasher40 = fast::sse2::Hh40;

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) type Neon8Hasher16 = fast::neon::Hh16;
#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) type Neon8Hasher24 = fast::neon::Hh24;
#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) type Neon8Hasher32 = fast::neon::Hh32;
#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) type Neon8Hasher40 = fast::neon::Hh40;
