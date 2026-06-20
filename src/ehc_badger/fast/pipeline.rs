//! Fused SIMD pipelines (V4, b=8) shared across block widths.

/// Instantiates monolithic EHC + tree + finalizer for a [`Block`] with `BYTES = 64`.
macro_rules! define_fused_backend {
    ($name:ident, $block:ty, $feature:literal) => {
        pub(crate) mod $name {
            use core::mem::MaybeUninit;

            use crate::block::{Block, multiply_add};
            use crate::ehc_badger::{combine3, combine4, combine5, encode3, encode4, encode5};
            use crate::entropy::{self, Entropy};

            type B = $block;
            type Row3 = [B; 3];

            const MAX_STACK: usize = 9;
            const FANOUT: usize = 8;
            const IN_W: usize = 3;
            const MACRO_BYTES_HH16: usize = 6 * IN_W * B::BYTES;
            const MACRO_BYTES_HH24: usize = 7 * IN_W * B::BYTES;
            const MACRO_BYTES_HH32: usize = 7 * IN_W * B::BYTES;
            const MACRO_BYTES_HH40: usize = 5 * IN_W * B::BYTES;

            /// Scratch stack/io storage — caller must fully initialize before read.
            #[inline(always)]
            #[allow(clippy::uninit_assumed_init)]
            unsafe fn uninit<T>() -> T {
                // SAFETY: deliberate uninitialized buffer; every caller overwrites all
                // elements before any read (`ehc_base_*` loads, `upper_layer*`, etc.).
                // SAFETY: Scratch storage is fully initialized by the caller before any read.
                unsafe { MaybeUninit::uninit().assume_init() }
            }

            #[inline(always)]
            fn mix(accum: B, input: B, seed: B) -> B {
                let output = input.plus32(seed);
                let twin = output.right_shift32();
                multiply_add(accum, output, twin)
            }

            #[inline(always)]
            fn mix_one(accum: B, input: B, seed: u64) -> B {
                mix(accum, input, B::load_one(seed))
            }

            #[inline(always)]
            fn mix_none(input: B, seed: u64) -> B {
                let output = input.plus32(B::load_one(seed));
                let twin = output.right_shift32();
                output.times(twin)
            }

            #[inline(always)]
            fn load_seed_ptr(words: *const u64, index: usize) -> B {
                // SAFETY: `words` entropy slice is large enough for the indexed seed block.
                unsafe { B::load(words.add(index) as *const u8) }
            }

            #[inline(always)]
            unsafe fn ehc_base_hh24(input: *const u8, words: *const u64) -> [B; 3] {
                // SAFETY: Scratch storage is fully initialized by the caller before any read.
                let mut io: [Row3; 9] = unsafe { uninit() };
                // SAFETY: `input` addresses one macro block (`MACRO_BYTES_*`); `words` is the seed pointer.
                unsafe {
                    io[0][0] = B::load(input);
                    io[0][1] = B::load(input.add(64));
                    io[0][2] = B::load(input.add(128));
                    io[1][0] = B::load(input.add(192));
                    io[1][1] = B::load(input.add(256));
                    io[1][2] = B::load(input.add(320));
                    io[2][0] = B::load(input.add(384));
                    io[2][1] = B::load(input.add(448));
                    io[2][2] = B::load(input.add(512));
                    io[3][0] = B::load(input.add(576));
                    io[3][1] = B::load(input.add(640));
                    io[3][2] = B::load(input.add(704));
                    io[4][0] = B::load(input.add(768));
                    io[4][1] = B::load(input.add(832));
                    io[4][2] = B::load(input.add(896));
                    io[5][0] = B::load(input.add(960));
                    io[5][1] = B::load(input.add(1024));
                    io[5][2] = B::load(input.add(1088));
                    io[6][0] = B::load(input.add(1152));
                    io[6][1] = B::load(input.add(1216));
                    io[6][2] = B::load(input.add(1280));
                    encode3(&mut io);
                    let w = words;
                    let mut h0 = mix_none(io[0][0], *w);
                    let mut h1 = mix_none(io[1][0], *w.add(3));
                    let mut h2 = mix_none(io[2][0], *w.add(6));
                    let mut h3 = mix_none(io[3][0], *w.add(9));
                    let mut h4 = mix_none(io[4][0], *w.add(12));
                    let mut h5 = mix_none(io[5][0], *w.add(15));
                    let mut h6 = mix_none(io[6][0], *w.add(18));
                    let mut h7 = mix_none(io[7][0], *w.add(21));
                    let mut h8 = mix_none(io[8][0], *w.add(24));
                    h0 = mix_one(h0, io[0][1], *w.add(1));
                    h1 = mix_one(h1, io[1][1], *w.add(4));
                    h2 = mix_one(h2, io[2][1], *w.add(7));
                    h3 = mix_one(h3, io[3][1], *w.add(10));
                    h4 = mix_one(h4, io[4][1], *w.add(13));
                    h5 = mix_one(h5, io[5][1], *w.add(16));
                    h6 = mix_one(h6, io[6][1], *w.add(19));
                    h7 = mix_one(h7, io[7][1], *w.add(22));
                    h8 = mix_one(h8, io[8][1], *w.add(25));
                    h0 = mix_one(h0, io[0][2], *w.add(2));
                    h1 = mix_one(h1, io[1][2], *w.add(5));
                    h2 = mix_one(h2, io[2][2], *w.add(8));
                    h3 = mix_one(h3, io[3][2], *w.add(11));
                    h4 = mix_one(h4, io[4][2], *w.add(14));
                    h5 = mix_one(h5, io[5][2], *w.add(17));
                    h6 = mix_one(h6, io[6][2], *w.add(20));
                    h7 = mix_one(h7, io[7][2], *w.add(23));
                    h8 = mix_one(h8, io[8][2], *w.add(26));
                    combine3(&[h0, h1, h2, h3, h4, h5, h6, h7, h8])
                }
            }

            #[inline(always)]
            unsafe fn ehc_base_hh16(input: *const u8, words: *const u64) -> [B; 2] {
                // SAFETY: `input` addresses one macro block (`MACRO_BYTES_*`); `words` is the seed pointer.
                unsafe {
                    let w = words;
                    let r00 = B::load(input);
                    let r10 = B::load(input.add(192));
                    let r20 = B::load(input.add(384));
                    let r30 = B::load(input.add(576));
                    let r40 = B::load(input.add(768));
                    let r50 = B::load(input.add(960));
                    let p0 = r00.xor(r10).xor(r20).xor(r30).xor(r40).xor(r50);

                    let mut h0 = mix_none(r00, *w);
                    let mut h1 = mix_none(r10, *w.add(3));
                    let mut h2 = mix_none(r20, *w.add(6));
                    let mut h3 = mix_none(r30, *w.add(9));
                    let mut h4 = mix_none(r40, *w.add(12));
                    let mut h5 = mix_none(r50, *w.add(15));
                    let mut h6 = mix_none(p0, *w.add(18));

                    let r01 = B::load(input.add(64));
                    let r11 = B::load(input.add(256));
                    let r21 = B::load(input.add(448));
                    let r31 = B::load(input.add(640));
                    let r41 = B::load(input.add(832));
                    let r51 = B::load(input.add(1024));
                    let p1 = r01.xor(r11).xor(r21).xor(r31).xor(r41).xor(r51);

                    h0 = mix_one(h0, r01, *w.add(1));
                    h1 = mix_one(h1, r11, *w.add(4));
                    h2 = mix_one(h2, r21, *w.add(7));
                    h3 = mix_one(h3, r31, *w.add(10));
                    h4 = mix_one(h4, r41, *w.add(13));
                    h5 = mix_one(h5, r51, *w.add(16));
                    h6 = mix_one(h6, p1, *w.add(19));

                    let r02 = B::load(input.add(128));
                    let r12 = B::load(input.add(320));
                    let r22 = B::load(input.add(512));
                    let r32 = B::load(input.add(704));
                    let r42 = B::load(input.add(896));
                    let r52 = B::load(input.add(1088));
                    let p2 = r02.xor(r12).xor(r22).xor(r32).xor(r42).xor(r52);

                    h0 = mix_one(h0, r02, *w.add(2));
                    h1 = mix_one(h1, r12, *w.add(5));
                    h2 = mix_one(h2, r22, *w.add(8));
                    h3 = mix_one(h3, r32, *w.add(11));
                    h4 = mix_one(h4, r42, *w.add(14));
                    h5 = mix_one(h5, r52, *w.add(17));
                    h6 = mix_one(h6, p2, *w.add(20));

                    let o0 = h0
                        .plus(h2)
                        .plus(h3)
                        .plus(h4.shl::<1>())
                        .plus(h5)
                        .plus(h6.shl::<2>());
                    let o1 = h1
                        .plus(h2)
                        .plus(h3.shl::<1>())
                        .plus(h4)
                        .plus(h5.shl::<2>())
                        .plus(h6);
                    [o0, o1]
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn ehc_base_hh32(input: *const u8, words: *const u64) -> [B; 4] {
                // SAFETY: Scratch storage is fully initialized by the caller before any read.
                let mut io: [Row3; 10] = unsafe { uninit() };
                // SAFETY: `input` addresses one macro block (`MACRO_BYTES_*`); `words` is the seed pointer.
                unsafe {
                    io[0][0] = B::load(input);
                    io[0][1] = B::load(input.add(64));
                    io[0][2] = B::load(input.add(128));
                    io[1][0] = B::load(input.add(192));
                    io[1][1] = B::load(input.add(256));
                    io[1][2] = B::load(input.add(320));
                    io[2][0] = B::load(input.add(384));
                    io[2][1] = B::load(input.add(448));
                    io[2][2] = B::load(input.add(512));
                    io[3][0] = B::load(input.add(576));
                    io[3][1] = B::load(input.add(640));
                    io[3][2] = B::load(input.add(704));
                    io[4][0] = B::load(input.add(768));
                    io[4][1] = B::load(input.add(832));
                    io[4][2] = B::load(input.add(896));
                    io[5][0] = B::load(input.add(960));
                    io[5][1] = B::load(input.add(1024));
                    io[5][2] = B::load(input.add(1088));
                    io[6][0] = B::load(input.add(1152));
                    io[6][1] = B::load(input.add(1216));
                    io[6][2] = B::load(input.add(1280));
                    encode4(&mut io);
                    let w = words;
                    let mut h0 = mix_none(io[0][0], *w);
                    let mut h1 = mix_none(io[1][0], *w.add(3));
                    let mut h2 = mix_none(io[2][0], *w.add(6));
                    let mut h3 = mix_none(io[3][0], *w.add(9));
                    let mut h4 = mix_none(io[4][0], *w.add(12));
                    let mut h5 = mix_none(io[5][0], *w.add(15));
                    let mut h6 = mix_none(io[6][0], *w.add(18));
                    let mut h7 = mix_none(io[7][0], *w.add(21));
                    let mut h8 = mix_none(io[8][0], *w.add(24));
                    let mut h9 = mix_none(io[9][0], *w.add(27));
                    h0 = mix_one(h0, io[0][1], *w.add(1));
                    h1 = mix_one(h1, io[1][1], *w.add(4));
                    h2 = mix_one(h2, io[2][1], *w.add(7));
                    h3 = mix_one(h3, io[3][1], *w.add(10));
                    h4 = mix_one(h4, io[4][1], *w.add(13));
                    h5 = mix_one(h5, io[5][1], *w.add(16));
                    h6 = mix_one(h6, io[6][1], *w.add(19));
                    h7 = mix_one(h7, io[7][1], *w.add(22));
                    h8 = mix_one(h8, io[8][1], *w.add(25));
                    h9 = mix_one(h9, io[9][1], *w.add(28));
                    h0 = mix_one(h0, io[0][2], *w.add(2));
                    h1 = mix_one(h1, io[1][2], *w.add(5));
                    h2 = mix_one(h2, io[2][2], *w.add(8));
                    h3 = mix_one(h3, io[3][2], *w.add(11));
                    h4 = mix_one(h4, io[4][2], *w.add(14));
                    h5 = mix_one(h5, io[5][2], *w.add(17));
                    h6 = mix_one(h6, io[6][2], *w.add(20));
                    h7 = mix_one(h7, io[7][2], *w.add(23));
                    h8 = mix_one(h8, io[8][2], *w.add(26));
                    h9 = mix_one(h9, io[9][2], *w.add(29));
                    combine4(&[h0, h1, h2, h3, h4, h5, h6, h7, h8, h9])
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn ehc_base_hh40(input: *const u8, words: *const u64) -> [B; 5] {
                // SAFETY: Scratch storage is fully initialized by the caller before any read.
                let mut io: [Row3; 9] = unsafe { uninit() };
                // SAFETY: `input` addresses one macro block (`MACRO_BYTES_*`); `words` is the seed pointer.
                unsafe {
                    io[0][0] = B::load(input);
                    io[0][1] = B::load(input.add(64));
                    io[0][2] = B::load(input.add(128));
                    io[1][0] = B::load(input.add(192));
                    io[1][1] = B::load(input.add(256));
                    io[1][2] = B::load(input.add(320));
                    io[2][0] = B::load(input.add(384));
                    io[2][1] = B::load(input.add(448));
                    io[2][2] = B::load(input.add(512));
                    io[3][0] = B::load(input.add(576));
                    io[3][1] = B::load(input.add(640));
                    io[3][2] = B::load(input.add(704));
                    io[4][0] = B::load(input.add(768));
                    io[4][1] = B::load(input.add(832));
                    io[4][2] = B::load(input.add(896));
                    encode5(&mut io);
                    let w = words;
                    let mut h0 = mix_none(io[0][0], *w);
                    let mut h1 = mix_none(io[1][0], *w.add(3));
                    let mut h2 = mix_none(io[2][0], *w.add(6));
                    let mut h3 = mix_none(io[3][0], *w.add(9));
                    let mut h4 = mix_none(io[4][0], *w.add(12));
                    let mut h5 = mix_none(io[5][0], *w.add(15));
                    let mut h6 = mix_none(io[6][0], *w.add(18));
                    let mut h7 = mix_none(io[7][0], *w.add(21));
                    let mut h8 = mix_none(io[8][0], *w.add(24));
                    h0 = mix_one(h0, io[0][1], *w.add(1));
                    h1 = mix_one(h1, io[1][1], *w.add(4));
                    h2 = mix_one(h2, io[2][1], *w.add(7));
                    h3 = mix_one(h3, io[3][1], *w.add(10));
                    h4 = mix_one(h4, io[4][1], *w.add(13));
                    h5 = mix_one(h5, io[5][1], *w.add(16));
                    h6 = mix_one(h6, io[6][1], *w.add(19));
                    h7 = mix_one(h7, io[7][1], *w.add(22));
                    h8 = mix_one(h8, io[8][1], *w.add(25));
                    h0 = mix_one(h0, io[0][2], *w.add(2));
                    h1 = mix_one(h1, io[1][2], *w.add(5));
                    h2 = mix_one(h2, io[2][2], *w.add(8));
                    h3 = mix_one(h3, io[3][2], *w.add(11));
                    h4 = mix_one(h4, io[4][2], *w.add(14));
                    h5 = mix_one(h5, io[5][2], *w.add(17));
                    h6 = mix_one(h6, io[6][2], *w.add(20));
                    h7 = mix_one(h7, io[7][2], *w.add(23));
                    h8 = mix_one(h8, io[8][2], *w.add(26));
                    combine5(&[h0, h1, h2, h3, h4, h5, h6, h7, h8])
                }
            }

            macro_rules! upper_layer_lane {
                ($acc:ident, $children:ident, $lane:literal, $w:ident) => {
                    $acc = mix_one($acc, $children[1][$lane], *$w);
                    $acc = mix_one($acc, $children[2][$lane], *$w.add(1));
                    $acc = mix_one($acc, $children[3][$lane], *$w.add(2));
                    $acc = mix_one($acc, $children[4][$lane], *$w.add(3));
                    $acc = mix_one($acc, $children[5][$lane], *$w.add(4));
                    $acc = mix_one($acc, $children[6][$lane], *$w.add(5));
                    $acc = mix_one($acc, $children[7][$lane], *$w.add(6));
                };
            }

            #[inline(always)]
            unsafe fn upper_layer3(
                children: &[[B; 3]; FANOUT],
                level: usize,
                words: *const u64,
                output: &mut [B; 3],
            ) {
                let base = entropy::tree_entropy_level(9, IN_W, FANOUT, 3, level);
                // SAFETY: See function safety invariants; operands satisfy `Block` / pointer contracts.
                unsafe {
                    let w = words.add(base);
                    let mut acc0 = children[0][0];
                    upper_layer_lane!(acc0, children, 0, w);
                    output[0] = acc0;

                    let w1 = w.add(FANOUT - 1);
                    let mut acc1 = children[0][1];
                    upper_layer_lane!(acc1, children, 1, w1);
                    output[1] = acc1;

                    let w2 = w1.add(FANOUT - 1);
                    let mut acc2 = children[0][2];
                    upper_layer_lane!(acc2, children, 2, w2);
                    output[2] = acc2;
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn upper_layer4(
                children: &[[B; 4]; FANOUT],
                level: usize,
                words: *const u64,
                output: &mut [B; 4],
            ) {
                let base = entropy::tree_entropy_level(10, IN_W, FANOUT, 4, level);
                // SAFETY: See function safety invariants; operands satisfy `Block` / pointer contracts.
                unsafe {
                    let w = words.add(base);
                    let mut acc0 = children[0][0];
                    upper_layer_lane!(acc0, children, 0, w);
                    output[0] = acc0;

                    let w1 = w.add(FANOUT - 1);
                    let mut acc1 = children[0][1];
                    upper_layer_lane!(acc1, children, 1, w1);
                    output[1] = acc1;

                    let w2 = w1.add(FANOUT - 1);
                    let mut acc2 = children[0][2];
                    upper_layer_lane!(acc2, children, 2, w2);
                    output[2] = acc2;

                    let w3 = w2.add(FANOUT - 1);
                    let mut acc3 = children[0][3];
                    upper_layer_lane!(acc3, children, 3, w3);
                    output[3] = acc3;
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn upper_layer5(
                children: &[[B; 5]; FANOUT],
                level: usize,
                words: *const u64,
                output: &mut [B; 5],
            ) {
                let base = entropy::tree_entropy_level(9, IN_W, FANOUT, 5, level);
                // SAFETY: See function safety invariants; operands satisfy `Block` / pointer contracts.
                unsafe {
                    let w = words.add(base);
                    let mut acc0 = children[0][0];
                    upper_layer_lane!(acc0, children, 0, w);
                    output[0] = acc0;

                    let w1 = w.add(FANOUT - 1);
                    let mut acc1 = children[0][1];
                    upper_layer_lane!(acc1, children, 1, w1);
                    output[1] = acc1;

                    let w2 = w1.add(FANOUT - 1);
                    let mut acc2 = children[0][2];
                    upper_layer_lane!(acc2, children, 2, w2);
                    output[2] = acc2;

                    let w3 = w2.add(FANOUT - 1);
                    let mut acc3 = children[0][3];
                    upper_layer_lane!(acc3, children, 3, w3);
                    output[3] = acc3;

                    let w4 = w3.add(FANOUT - 1);
                    let mut acc4 = children[0][4];
                    upper_layer_lane!(acc4, children, 4, w4);
                    output[4] = acc4;
                }
            }

            #[inline(always)]
            unsafe fn upper_layer2(
                children: &[[B; 2]; FANOUT],
                level: usize,
                words: *const u64,
                output: &mut [B; 2],
            ) {
                let base = entropy::tree_entropy_level(7, IN_W, FANOUT, 2, level);
                // SAFETY: See function safety invariants; operands satisfy `Block` / pointer contracts.
                unsafe {
                    let w = words.add(base);
                    let mut acc0 = children[0][0];
                    upper_layer_lane!(acc0, children, 0, w);
                    output[0] = acc0;

                    let w1 = w.add(FANOUT - 1);
                    let mut acc1 = children[0][1];
                    upper_layer_lane!(acc1, children, 1, w1);
                    output[1] = acc1;
                }
            }

            /// Reference `V4Avx2::Hash` — local stack, no intermediate struct writes.
            #[target_feature(enable = $feature)]
            unsafe fn hash_v4_hh24(words: &[u64], data: *const u8, len: usize) -> [u64; 3] {
                // SAFETY: `data`/`tail` span `len`/`tail_len` bytes; entropy slice outlives the hash.
                unsafe {
                    let words = words.as_ptr();
                    let mut stack: [[[B; 3]; FANOUT]; MAX_STACK] = uninit();
                    let mut stack_lengths = [0u8; MAX_STACK];
                    let block_count = len / MACRO_BYTES_HH24;

                    for k in 0..block_count {
                        if stack_lengths[0] < FANOUT as u8 {
                            let slot = stack_lengths[0] as usize;
                            stack[0][slot] = ehc_base_hh24(data.add(k * MACRO_BYTES_HH24), words);
                            stack_lengths[0] += 1;
                            continue;
                        }
                        let mut i = 0usize;
                        while i < MAX_STACK && stack_lengths[i] as usize == FANOUT {
                            i += 1;
                        }
                        for j in (0..i).rev() {
                            let mut combined: [B; 3] = uninit();
                            upper_layer3(&stack[j], j, words, &mut combined);
                            let slot = stack_lengths[j + 1] as usize;
                            stack[j + 1][slot] = combined;
                            stack_lengths[j] = 0;
                            stack_lengths[j + 1] += 1;
                        }
                        let slot = stack_lengths[0] as usize;
                        stack[0][slot] = ehc_base_hh24(data.add(k * MACRO_BYTES_HH24), words);
                        stack_lengths[0] += 1;
                    }

                    let fin_base = entropy::finalizer_entropy_base(9, IN_W, FANOUT, 3);
                    let tail = data.add(block_count * MACRO_BYTES_HH24);
                    let tail_len = len - block_count * MACRO_BYTES_HH24;
                    let mut accum = [B::zero(), B::zero(), B::zero()];
                    let mut seed_index = fin_base;

                    for level in 0..MAX_STACK {
                        for slot in 0..stack_lengths[level] as usize {
                            let chunk = stack[level][slot];
                            accum[0] = mix(accum[0], chunk[0], load_seed_ptr(words, seed_index));
                            accum[1] = mix(
                                accum[1],
                                chunk[1],
                                load_seed_ptr(words, seed_index + B::LANES),
                            );
                            accum[2] = mix(
                                accum[2],
                                chunk[2],
                                load_seed_ptr(words, seed_index + 2 * B::LANES),
                            );
                            seed_index += 3 * B::LANES;
                        }
                    }

                    let mut i = 0usize;
                    while i + B::BYTES <= tail_len {
                        let word = B::load(tail.add(i));
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                        seed_index += B::LANES;
                        i += B::BYTES;
                    }
                    {
                        let mut extra = [0u8; 64];
                        if i < tail_len {
                            extra[..tail_len - i].copy_from_slice(core::slice::from_raw_parts(
                                tail.add(i),
                                tail_len - i,
                            ));
                        }
                        let word = B::load(extra.as_ptr());
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                    }

                    [accum[0].sum(), accum[1].sum(), accum[2].sum()]
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn hash_v4_hh16(words: &[u64], data: *const u8, len: usize) -> [u64; 2] {
                // SAFETY: `data`/`tail` span `len`/`tail_len` bytes; entropy slice outlives the hash.
                unsafe {
                    let words = words.as_ptr();
                    let mut stack: [[[B; 2]; FANOUT]; MAX_STACK] = uninit();
                    let mut stack_lengths = [0u8; MAX_STACK];
                    let block_count = len / MACRO_BYTES_HH16;

                    for k in 0..block_count {
                        if stack_lengths[0] < FANOUT as u8 {
                            let slot = stack_lengths[0] as usize;
                            stack[0][slot] = ehc_base_hh16(data.add(k * MACRO_BYTES_HH16), words);
                            stack_lengths[0] += 1;
                            continue;
                        }
                        let mut i = 0usize;
                        while i < MAX_STACK && stack_lengths[i] as usize == FANOUT {
                            i += 1;
                        }
                        for j in (0..i).rev() {
                            let mut combined: [B; 2] = uninit();
                            upper_layer2(&stack[j], j, words, &mut combined);
                            let slot = stack_lengths[j + 1] as usize;
                            stack[j + 1][slot] = combined;
                            stack_lengths[j] = 0;
                            stack_lengths[j + 1] += 1;
                        }
                        let slot = stack_lengths[0] as usize;
                        stack[0][slot] = ehc_base_hh16(data.add(k * MACRO_BYTES_HH16), words);
                        stack_lengths[0] += 1;
                    }

                    let fin_base = entropy::finalizer_entropy_base(7, IN_W, FANOUT, 2);
                    let tail = data.add(block_count * MACRO_BYTES_HH16);
                    let tail_len = len - block_count * MACRO_BYTES_HH16;
                    let mut accum = [B::zero(), B::zero()];
                    let mut seed_index = fin_base;

                    for level in 0..MAX_STACK {
                        for slot in 0..stack_lengths[level] as usize {
                            let chunk = stack[level][slot];
                            accum[0] = mix(accum[0], chunk[0], load_seed_ptr(words, seed_index));
                            accum[1] = mix(
                                accum[1],
                                chunk[1],
                                load_seed_ptr(words, seed_index + B::LANES),
                            );
                            seed_index += 2 * B::LANES;
                        }
                    }

                    let mut i = 0usize;
                    while i + B::BYTES <= tail_len {
                        let word = B::load(tail.add(i));
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        seed_index += B::LANES;
                        i += B::BYTES;
                    }
                    {
                        let mut extra = [0u8; 64];
                        if i < tail_len {
                            extra[..tail_len - i].copy_from_slice(core::slice::from_raw_parts(
                                tail.add(i),
                                tail_len - i,
                            ));
                        }
                        let word = B::load(extra.as_ptr());
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                    }

                    [accum[0].sum(), accum[1].sum()]
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn hash_v4_hh32(words: &[u64], data: *const u8, len: usize) -> [u64; 4] {
                // SAFETY: `data`/`tail` span `len`/`tail_len` bytes; entropy slice outlives the hash.
                unsafe {
                    let words = words.as_ptr();
                    let mut stack: [[[B; 4]; FANOUT]; MAX_STACK] = uninit();
                    let mut stack_lengths = [0u8; MAX_STACK];
                    let block_count = len / MACRO_BYTES_HH32;

                    for k in 0..block_count {
                        if stack_lengths[0] < FANOUT as u8 {
                            let slot = stack_lengths[0] as usize;
                            stack[0][slot] = ehc_base_hh32(data.add(k * MACRO_BYTES_HH32), words);
                            stack_lengths[0] += 1;
                            continue;
                        }
                        let mut i = 0usize;
                        while i < MAX_STACK && stack_lengths[i] as usize == FANOUT {
                            i += 1;
                        }
                        for j in (0..i).rev() {
                            let mut combined: [B; 4] = uninit();
                            upper_layer4(&stack[j], j, words, &mut combined);
                            let slot = stack_lengths[j + 1] as usize;
                            stack[j + 1][slot] = combined;
                            stack_lengths[j] = 0;
                            stack_lengths[j + 1] += 1;
                        }
                        let slot = stack_lengths[0] as usize;
                        stack[0][slot] = ehc_base_hh32(data.add(k * MACRO_BYTES_HH32), words);
                        stack_lengths[0] += 1;
                    }

                    let fin_base = entropy::finalizer_entropy_base(10, IN_W, FANOUT, 4);
                    let tail = data.add(block_count * MACRO_BYTES_HH32);
                    let tail_len = len - block_count * MACRO_BYTES_HH32;
                    let mut accum = [B::zero(), B::zero(), B::zero(), B::zero()];
                    let mut seed_index = fin_base;

                    for level in 0..MAX_STACK {
                        for slot in 0..stack_lengths[level] as usize {
                            let chunk = stack[level][slot];
                            accum[0] = mix(accum[0], chunk[0], load_seed_ptr(words, seed_index));
                            accum[1] = mix(
                                accum[1],
                                chunk[1],
                                load_seed_ptr(words, seed_index + B::LANES),
                            );
                            accum[2] = mix(
                                accum[2],
                                chunk[2],
                                load_seed_ptr(words, seed_index + 2 * B::LANES),
                            );
                            accum[3] = mix(
                                accum[3],
                                chunk[3],
                                load_seed_ptr(words, seed_index + 3 * B::LANES),
                            );
                            seed_index += 4 * B::LANES;
                        }
                    }

                    let mut i = 0usize;
                    while i + B::BYTES <= tail_len {
                        let word = B::load(tail.add(i));
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                        accum[3] = mix(
                            accum[3],
                            word,
                            load_seed_ptr(words, seed_index + 3 * B::LANES),
                        );
                        seed_index += B::LANES;
                        i += B::BYTES;
                    }
                    {
                        let mut extra = [0u8; 64];
                        if i < tail_len {
                            extra[..tail_len - i].copy_from_slice(core::slice::from_raw_parts(
                                tail.add(i),
                                tail_len - i,
                            ));
                        }
                        let word = B::load(extra.as_ptr());
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                        accum[3] = mix(
                            accum[3],
                            word,
                            load_seed_ptr(words, seed_index + 3 * B::LANES),
                        );
                    }

                    [
                        accum[0].sum(),
                        accum[1].sum(),
                        accum[2].sum(),
                        accum[3].sum(),
                    ]
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn hash_v4_hh40(words: &[u64], data: *const u8, len: usize) -> [u64; 5] {
                // SAFETY: `data`/`tail` span `len`/`tail_len` bytes; entropy slice outlives the hash.
                unsafe {
                    let words = words.as_ptr();
                    let mut stack: [[[B; 5]; FANOUT]; MAX_STACK] = uninit();
                    let mut stack_lengths = [0u8; MAX_STACK];
                    let block_count = len / MACRO_BYTES_HH40;

                    for k in 0..block_count {
                        if stack_lengths[0] < FANOUT as u8 {
                            let slot = stack_lengths[0] as usize;
                            stack[0][slot] = ehc_base_hh40(data.add(k * MACRO_BYTES_HH40), words);
                            stack_lengths[0] += 1;
                            continue;
                        }
                        let mut i = 0usize;
                        while i < MAX_STACK && stack_lengths[i] as usize == FANOUT {
                            i += 1;
                        }
                        for j in (0..i).rev() {
                            let mut combined: [B; 5] = uninit();
                            upper_layer5(&stack[j], j, words, &mut combined);
                            let slot = stack_lengths[j + 1] as usize;
                            stack[j + 1][slot] = combined;
                            stack_lengths[j] = 0;
                            stack_lengths[j + 1] += 1;
                        }
                        let slot = stack_lengths[0] as usize;
                        stack[0][slot] = ehc_base_hh40(data.add(k * MACRO_BYTES_HH40), words);
                        stack_lengths[0] += 1;
                    }

                    let fin_base = entropy::finalizer_entropy_base(9, IN_W, FANOUT, 5);
                    let tail = data.add(block_count * MACRO_BYTES_HH40);
                    let tail_len = len - block_count * MACRO_BYTES_HH40;
                    let mut accum = [B::zero(), B::zero(), B::zero(), B::zero(), B::zero()];
                    let mut seed_index = fin_base;

                    for level in 0..MAX_STACK {
                        for slot in 0..stack_lengths[level] as usize {
                            let chunk = stack[level][slot];
                            accum[0] = mix(accum[0], chunk[0], load_seed_ptr(words, seed_index));
                            accum[1] = mix(
                                accum[1],
                                chunk[1],
                                load_seed_ptr(words, seed_index + B::LANES),
                            );
                            accum[2] = mix(
                                accum[2],
                                chunk[2],
                                load_seed_ptr(words, seed_index + 2 * B::LANES),
                            );
                            accum[3] = mix(
                                accum[3],
                                chunk[3],
                                load_seed_ptr(words, seed_index + 3 * B::LANES),
                            );
                            accum[4] = mix(
                                accum[4],
                                chunk[4],
                                load_seed_ptr(words, seed_index + 4 * B::LANES),
                            );
                            seed_index += 5 * B::LANES;
                        }
                    }

                    let mut i = 0usize;
                    while i + B::BYTES <= tail_len {
                        let word = B::load(tail.add(i));
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                        accum[3] = mix(
                            accum[3],
                            word,
                            load_seed_ptr(words, seed_index + 3 * B::LANES),
                        );
                        accum[4] = mix(
                            accum[4],
                            word,
                            load_seed_ptr(words, seed_index + 4 * B::LANES),
                        );
                        seed_index += B::LANES;
                        i += B::BYTES;
                    }
                    {
                        let mut extra = [0u8; 64];
                        if i < tail_len {
                            extra[..tail_len - i].copy_from_slice(core::slice::from_raw_parts(
                                tail.add(i),
                                tail_len - i,
                            ));
                        }
                        let word = B::load(extra.as_ptr());
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                        accum[3] = mix(
                            accum[3],
                            word,
                            load_seed_ptr(words, seed_index + 3 * B::LANES),
                        );
                        accum[4] = mix(
                            accum[4],
                            word,
                            load_seed_ptr(words, seed_index + 4 * B::LANES),
                        );
                    }

                    [
                        accum[0].sum(),
                        accum[1].sum(),
                        accum[2].sum(),
                        accum[3].sum(),
                        accum[4].sum(),
                    ]
                }
            }

            #[inline(always)]
            unsafe fn dfs_tree_hash24(
                stack: &mut [[[B; 3]; FANOUT]; MAX_STACK],
                stack_lengths: &mut [u8; MAX_STACK],
                data: *const u8,
                block_count: usize,
                words: &[u64],
            ) {
                for k in 0..block_count {
                    if stack_lengths[0] < FANOUT as u8 {
                        let slot = stack_lengths[0] as usize;
                        // SAFETY: `data` points at one full macro block; `words` is the entropy seed pointer.
                        stack[0][slot] = unsafe {
                            ehc_base_hh24(data.add(k * MACRO_BYTES_HH24), words.as_ptr())
                        };
                        stack_lengths[0] += 1;
                        continue;
                    }
                    let mut i = 0usize;
                    while i < MAX_STACK && stack_lengths[i] as usize == FANOUT {
                        i += 1;
                    }
                    for j in (0..i).rev() {
                        // SAFETY: Scratch storage is fully initialized by the caller before any read.
                        let mut combined: [B; 3] = unsafe { uninit() };
                        // SAFETY: `words` covers tree entropy at `level`; `combined` is fully written.
                        unsafe { upper_layer3(&stack[j], j, words.as_ptr(), &mut combined) };
                        let slot = stack_lengths[j + 1] as usize;
                        stack[j + 1][slot] = combined;
                        stack_lengths[j] = 0;
                        stack_lengths[j + 1] += 1;
                    }
                    let slot = stack_lengths[0] as usize;
                    // SAFETY: `data` points at one full macro block; `words` is the entropy seed pointer.
                    unsafe {
                        stack[0][slot] =
                            ehc_base_hh24(data.add(k * MACRO_BYTES_HH24), words.as_ptr());
                    }
                    stack_lengths[0] += 1;
                }
            }

            #[inline(always)]
            unsafe fn dfs_tree_hash16(
                stack: &mut [[[B; 2]; FANOUT]; MAX_STACK],
                stack_lengths: &mut [u8; MAX_STACK],
                data: *const u8,
                block_count: usize,
                words: &[u64],
            ) {
                for k in 0..block_count {
                    if stack_lengths[0] < FANOUT as u8 {
                        let slot = stack_lengths[0] as usize;
                        // SAFETY: `data` points at one full macro block; `words` is the entropy seed pointer.
                        stack[0][slot] = unsafe {
                            ehc_base_hh16(data.add(k * MACRO_BYTES_HH16), words.as_ptr())
                        };
                        stack_lengths[0] += 1;
                        continue;
                    }
                    let mut i = 0usize;
                    while i < MAX_STACK && stack_lengths[i] as usize == FANOUT {
                        i += 1;
                    }
                    for j in (0..i).rev() {
                        // SAFETY: Scratch storage is fully initialized by the caller before any read.
                        let mut combined: [B; 2] = unsafe { uninit() };
                        // SAFETY: `words` covers tree entropy at `level`; `combined` is fully written.
                        unsafe { upper_layer2(&stack[j], j, words.as_ptr(), &mut combined) };
                        let slot = stack_lengths[j + 1] as usize;
                        stack[j + 1][slot] = combined;
                        stack_lengths[j] = 0;
                        stack_lengths[j + 1] += 1;
                    }
                    let slot = stack_lengths[0] as usize;
                    // SAFETY: `data` points at one full macro block; `words` is the entropy seed pointer.
                    unsafe {
                        stack[0][slot] =
                            ehc_base_hh16(data.add(k * MACRO_BYTES_HH16), words.as_ptr());
                    }
                    stack_lengths[0] += 1;
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn dfs_tree_hash32(
                stack: &mut [[[B; 4]; FANOUT]; MAX_STACK],
                stack_lengths: &mut [u8; MAX_STACK],
                data: *const u8,
                block_count: usize,
                words: &[u64],
            ) {
                for k in 0..block_count {
                    if stack_lengths[0] < FANOUT as u8 {
                        let slot = stack_lengths[0] as usize;
                        // SAFETY: `data` points at one full macro block; `words` is the entropy seed pointer.
                        stack[0][slot] = unsafe {
                            ehc_base_hh32(data.add(k * MACRO_BYTES_HH32), words.as_ptr())
                        };
                        stack_lengths[0] += 1;
                        continue;
                    }
                    let mut i = 0usize;
                    while i < MAX_STACK && stack_lengths[i] as usize == FANOUT {
                        i += 1;
                    }
                    for j in (0..i).rev() {
                        // SAFETY: Scratch storage is fully initialized by the caller before any read.
                        let mut combined: [B; 4] = unsafe { uninit() };
                        // SAFETY: `words` covers tree entropy at `level`; `combined` is fully written.
                        unsafe { upper_layer4(&stack[j], j, words.as_ptr(), &mut combined) };
                        let slot = stack_lengths[j + 1] as usize;
                        stack[j + 1][slot] = combined;
                        stack_lengths[j] = 0;
                        stack_lengths[j + 1] += 1;
                    }
                    let slot = stack_lengths[0] as usize;
                    // SAFETY: `data` points at one full macro block; `words` is the entropy seed pointer.
                    unsafe {
                        stack[0][slot] =
                            ehc_base_hh32(data.add(k * MACRO_BYTES_HH32), words.as_ptr());
                    }
                    stack_lengths[0] += 1;
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn dfs_tree_hash40(
                stack: &mut [[[B; 5]; FANOUT]; MAX_STACK],
                stack_lengths: &mut [u8; MAX_STACK],
                data: *const u8,
                block_count: usize,
                words: &[u64],
            ) {
                for k in 0..block_count {
                    if stack_lengths[0] < FANOUT as u8 {
                        let slot = stack_lengths[0] as usize;
                        // SAFETY: `data` points at one full macro block; `words` is the entropy seed pointer.
                        stack[0][slot] = unsafe {
                            ehc_base_hh40(data.add(k * MACRO_BYTES_HH40), words.as_ptr())
                        };
                        stack_lengths[0] += 1;
                        continue;
                    }
                    let mut i = 0usize;
                    while i < MAX_STACK && stack_lengths[i] as usize == FANOUT {
                        i += 1;
                    }
                    for j in (0..i).rev() {
                        // SAFETY: Scratch storage is fully initialized by the caller before any read.
                        let mut combined: [B; 5] = unsafe { uninit() };
                        // SAFETY: `words` covers tree entropy at `level`; `combined` is fully written.
                        unsafe { upper_layer5(&stack[j], j, words.as_ptr(), &mut combined) };
                        let slot = stack_lengths[j + 1] as usize;
                        stack[j + 1][slot] = combined;
                        stack_lengths[j] = 0;
                        stack_lengths[j + 1] += 1;
                    }
                    let slot = stack_lengths[0] as usize;
                    // SAFETY: `data` points at one full macro block; `words` is the entropy seed pointer.
                    unsafe {
                        stack[0][slot] =
                            ehc_base_hh40(data.add(k * MACRO_BYTES_HH40), words.as_ptr());
                    }
                    stack_lengths[0] += 1;
                }
            }

            #[inline(always)]
            unsafe fn greedy_finish3(
                stack: &[[[B; 3]; FANOUT]; MAX_STACK],
                stack_lengths: &[u8; MAX_STACK],
                tail: *const u8,
                tail_len: usize,
                words: &[u64],
                fin_base: usize,
            ) -> [u64; 3] {
                // SAFETY: `tail` spans `tail_len` bytes; entropy slice outlives the hash.
                unsafe {
                    let words = words.as_ptr();
                    let mut accum = [B::zero(), B::zero(), B::zero()];
                    let mut seed_index = fin_base;
                    for level in 0..MAX_STACK {
                        for slot in 0..stack_lengths[level] as usize {
                            let chunk = stack[level][slot];
                            accum[0] = mix(accum[0], chunk[0], load_seed_ptr(words, seed_index));
                            accum[1] = mix(
                                accum[1],
                                chunk[1],
                                load_seed_ptr(words, seed_index + B::LANES),
                            );
                            accum[2] = mix(
                                accum[2],
                                chunk[2],
                                load_seed_ptr(words, seed_index + 2 * B::LANES),
                            );
                            seed_index += 3 * B::LANES;
                        }
                    }
                    let mut i = 0usize;
                    while i + B::BYTES <= tail_len {
                        let word = B::load(tail.add(i));
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                        seed_index += B::LANES;
                        i += B::BYTES;
                    }
                    {
                        let mut extra = [0u8; 64];
                        if i < tail_len {
                            extra[..tail_len - i].copy_from_slice(core::slice::from_raw_parts(
                                tail.add(i),
                                tail_len - i,
                            ));
                        }
                        let word = B::load(extra.as_ptr());
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                    }
                    [accum[0].sum(), accum[1].sum(), accum[2].sum()]
                }
            }

            #[inline(always)]
            unsafe fn greedy_finish2(
                stack: &[[[B; 2]; FANOUT]; MAX_STACK],
                stack_lengths: &[u8; MAX_STACK],
                tail: *const u8,
                tail_len: usize,
                words: &[u64],
                fin_base: usize,
            ) -> [u64; 2] {
                // SAFETY: `tail` spans `tail_len` bytes; entropy slice outlives the hash.
                unsafe {
                    let words = words.as_ptr();
                    let mut accum = [B::zero(), B::zero()];
                    let mut seed_index = fin_base;
                    for level in 0..MAX_STACK {
                        for slot in 0..stack_lengths[level] as usize {
                            let chunk = stack[level][slot];
                            accum[0] = mix(accum[0], chunk[0], load_seed_ptr(words, seed_index));
                            accum[1] = mix(
                                accum[1],
                                chunk[1],
                                load_seed_ptr(words, seed_index + B::LANES),
                            );
                            seed_index += 2 * B::LANES;
                        }
                    }
                    let mut i = 0usize;
                    while i + B::BYTES <= tail_len {
                        let word = B::load(tail.add(i));
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        seed_index += B::LANES;
                        i += B::BYTES;
                    }
                    {
                        let mut extra = [0u8; 64];
                        if i < tail_len {
                            extra[..tail_len - i].copy_from_slice(core::slice::from_raw_parts(
                                tail.add(i),
                                tail_len - i,
                            ));
                        }
                        let word = B::load(extra.as_ptr());
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                    }
                    [accum[0].sum(), accum[1].sum()]
                }
            }

            #[inline(always)]
            unsafe fn greedy_finish4(
                stack: &[[[B; 4]; FANOUT]; MAX_STACK],
                stack_lengths: &[u8; MAX_STACK],
                tail: *const u8,
                tail_len: usize,
                words: &[u64],
                fin_base: usize,
            ) -> [u64; 4] {
                // SAFETY: `tail` spans `tail_len` bytes; entropy slice outlives the hash.
                unsafe {
                    let words = words.as_ptr();
                    let mut accum = [B::zero(), B::zero(), B::zero(), B::zero()];
                    let mut seed_index = fin_base;
                    for level in 0..MAX_STACK {
                        for slot in 0..stack_lengths[level] as usize {
                            let chunk = stack[level][slot];
                            accum[0] = mix(accum[0], chunk[0], load_seed_ptr(words, seed_index));
                            accum[1] = mix(
                                accum[1],
                                chunk[1],
                                load_seed_ptr(words, seed_index + B::LANES),
                            );
                            accum[2] = mix(
                                accum[2],
                                chunk[2],
                                load_seed_ptr(words, seed_index + 2 * B::LANES),
                            );
                            accum[3] = mix(
                                accum[3],
                                chunk[3],
                                load_seed_ptr(words, seed_index + 3 * B::LANES),
                            );
                            seed_index += 4 * B::LANES;
                        }
                    }
                    let mut i = 0usize;
                    while i + B::BYTES <= tail_len {
                        let word = B::load(tail.add(i));
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                        accum[3] = mix(
                            accum[3],
                            word,
                            load_seed_ptr(words, seed_index + 3 * B::LANES),
                        );
                        seed_index += B::LANES;
                        i += B::BYTES;
                    }
                    {
                        let mut extra = [0u8; 64];
                        if i < tail_len {
                            extra[..tail_len - i].copy_from_slice(core::slice::from_raw_parts(
                                tail.add(i),
                                tail_len - i,
                            ));
                        }
                        let word = B::load(extra.as_ptr());
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                        accum[3] = mix(
                            accum[3],
                            word,
                            load_seed_ptr(words, seed_index + 3 * B::LANES),
                        );
                    }
                    [
                        accum[0].sum(),
                        accum[1].sum(),
                        accum[2].sum(),
                        accum[3].sum(),
                    ]
                }
            }

            #[inline(always)]
            unsafe fn greedy_finish5(
                stack: &[[[B; 5]; FANOUT]; MAX_STACK],
                stack_lengths: &[u8; MAX_STACK],
                tail: *const u8,
                tail_len: usize,
                words: &[u64],
                fin_base: usize,
            ) -> [u64; 5] {
                // SAFETY: `tail` spans `tail_len` bytes; entropy slice outlives the hash.
                unsafe {
                    let words = words.as_ptr();
                    let mut accum = [B::zero(), B::zero(), B::zero(), B::zero(), B::zero()];
                    let mut seed_index = fin_base;
                    for level in 0..MAX_STACK {
                        for slot in 0..stack_lengths[level] as usize {
                            let chunk = stack[level][slot];
                            accum[0] = mix(accum[0], chunk[0], load_seed_ptr(words, seed_index));
                            accum[1] = mix(
                                accum[1],
                                chunk[1],
                                load_seed_ptr(words, seed_index + B::LANES),
                            );
                            accum[2] = mix(
                                accum[2],
                                chunk[2],
                                load_seed_ptr(words, seed_index + 2 * B::LANES),
                            );
                            accum[3] = mix(
                                accum[3],
                                chunk[3],
                                load_seed_ptr(words, seed_index + 3 * B::LANES),
                            );
                            accum[4] = mix(
                                accum[4],
                                chunk[4],
                                load_seed_ptr(words, seed_index + 4 * B::LANES),
                            );
                            seed_index += 5 * B::LANES;
                        }
                    }
                    let mut i = 0usize;
                    while i + B::BYTES <= tail_len {
                        let word = B::load(tail.add(i));
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                        accum[3] = mix(
                            accum[3],
                            word,
                            load_seed_ptr(words, seed_index + 3 * B::LANES),
                        );
                        accum[4] = mix(
                            accum[4],
                            word,
                            load_seed_ptr(words, seed_index + 4 * B::LANES),
                        );
                        seed_index += B::LANES;
                        i += B::BYTES;
                    }
                    {
                        let mut extra = [0u8; 64];
                        if i < tail_len {
                            extra[..tail_len - i].copy_from_slice(core::slice::from_raw_parts(
                                tail.add(i),
                                tail_len - i,
                            ));
                        }
                        let word = B::load(extra.as_ptr());
                        accum[0] = mix(accum[0], word, load_seed_ptr(words, seed_index));
                        accum[1] = mix(accum[1], word, load_seed_ptr(words, seed_index + B::LANES));
                        accum[2] = mix(
                            accum[2],
                            word,
                            load_seed_ptr(words, seed_index + 2 * B::LANES),
                        );
                        accum[3] = mix(
                            accum[3],
                            word,
                            load_seed_ptr(words, seed_index + 3 * B::LANES),
                        );
                        accum[4] = mix(
                            accum[4],
                            word,
                            load_seed_ptr(words, seed_index + 4 * B::LANES),
                        );
                    }
                    [
                        accum[0].sum(),
                        accum[1].sum(),
                        accum[2].sum(),
                        accum[3].sum(),
                        accum[4].sum(),
                    ]
                }
            }

            fn stack_is_empty(lengths: &[u8; MAX_STACK], buffer_len: usize) -> bool {
                buffer_len == 0 && lengths.iter().all(|&l| l == 0)
            }

            /// Fast `Hasher` for `HalftimeHash24` / AVX2×2.
            #[derive(Clone)]
            pub(crate) struct Hh24 {
                stack: [[[B; 3]; FANOUT]; MAX_STACK],
                stack_lengths: [u8; MAX_STACK],
                buffer: [u8; 2048],
                buffer_len: usize,
                total_len: u64,
                entropy: Entropy,
                /// First `update` on empty state: defer to `hash_v4` in `finalize`.
                pending: Option<(*const u8, usize)>,
            }

            impl Hh24 {
                pub(crate) const INSTANCE_BYTES: usize = 7 * IN_W * 8;

                pub(crate) fn new(key: &[u8; 32]) -> Self {
                    Self::from_prepared_entropy(entropy::entropy_for_key(key, B::LANES, 7, 9, 3))
                }

                pub(crate) fn from_prepared_entropy(entropy: Entropy) -> Self {
                    Self {
                        stack: core::array::from_fn(|_| [[B::load_one(0); 3]; FANOUT]),
                        stack_lengths: [0; MAX_STACK],
                        buffer: [0; 2048],
                        buffer_len: 0,
                        total_len: 0,
                        entropy,
                        pending: None,
                    }
                }

                #[target_feature(enable = $feature)]
                unsafe fn flush_pending(&mut self) {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        if let Some((ptr, len)) = self.pending.take() {
                            let words = self.entropy.as_slice();
                            let block_count = len / MACRO_BYTES_HH24;
                            dfs_tree_hash24(
                                &mut self.stack,
                                &mut self.stack_lengths,
                                ptr,
                                block_count,
                                words,
                            );
                            let tail_off = block_count * MACRO_BYTES_HH24;
                            let tail_len = len - tail_off;
                            if tail_len > 0 {
                                self.buffer[..tail_len].copy_from_slice(
                                    core::slice::from_raw_parts(ptr.add(tail_off), tail_len),
                                );
                            }
                            self.buffer_len = tail_len;
                        }
                    }
                }

                pub(crate) fn update(&mut self, data: &[u8]) {
                    // SAFETY: Backend `update`/`finalize` requires the matching CPU feature to be enabled.
                    unsafe { self.update_impl(data) }
                }

                #[target_feature(enable = $feature)]
                unsafe fn update_impl(&mut self, data: &[u8]) {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        self.total_len = self.total_len.wrapping_add(data.len() as u64);

                        if self.pending.is_some() {
                            self.flush_pending();
                        }

                        if stack_is_empty(&self.stack_lengths, self.buffer_len) {
                            self.pending = Some((data.as_ptr(), data.len()));
                            return;
                        }

                        let words = self.entropy.as_slice();
                        let mut off = 0usize;
                        if self.buffer_len > 0 {
                            let need = Self::INSTANCE_BYTES - self.buffer_len;
                            let take = need.min(data.len());
                            self.buffer[self.buffer_len..self.buffer_len + take]
                                .copy_from_slice(&data[..take]);
                            self.buffer_len += take;
                            off += take;
                            if self.buffer_len == Self::INSTANCE_BYTES {
                                let mut padded = [0u8; MACRO_BYTES_HH24];
                                padded[..Self::INSTANCE_BYTES]
                                    .copy_from_slice(&self.buffer[..Self::INSTANCE_BYTES]);
                                dfs_tree_hash24(
                                    &mut self.stack,
                                    &mut self.stack_lengths,
                                    padded.as_ptr(),
                                    1,
                                    words,
                                );
                                self.buffer_len = 0;
                            }
                        }

                        let block_count = (data.len() - off) / MACRO_BYTES_HH24;
                        if block_count > 0 {
                            dfs_tree_hash24(
                                &mut self.stack,
                                &mut self.stack_lengths,
                                data.as_ptr().add(off),
                                block_count,
                                words,
                            );
                            off += block_count * MACRO_BYTES_HH24;
                        }

                        if off < data.len() {
                            let rem = data.len() - off;
                            self.buffer[..rem].copy_from_slice(&data[off..]);
                            self.buffer_len = rem;
                        }
                    }
                }

                pub(crate) fn finalize(self) -> [u64; 3] {
                    // SAFETY: Backend `update`/`finalize` requires the matching CPU feature to be enabled.
                    unsafe { self.finalize_impl() }
                }

                #[target_feature(enable = $feature)]
                unsafe fn finalize_impl(self) -> [u64; 3] {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        if let Some((ptr, len)) = self.pending {
                            return hash_v4_hh24(self.entropy.as_slice(), ptr, len);
                        }
                        let fin_base = entropy::finalizer_entropy_base(9, IN_W, FANOUT, 3);
                        greedy_finish3(
                            &self.stack,
                            &self.stack_lengths,
                            self.buffer.as_ptr(),
                            self.buffer_len,
                            self.entropy.as_slice(),
                            fin_base,
                        )
                    }
                }
            }

            /// Fast `Hasher` for `HalftimeHash16` / AVX2×2.
            #[derive(Clone)]
            pub(crate) struct Hh16 {
                stack: [[[B; 2]; FANOUT]; MAX_STACK],
                stack_lengths: [u8; MAX_STACK],
                buffer: [u8; 2048],
                buffer_len: usize,
                total_len: u64,
                entropy: Entropy,
                pending: Option<(*const u8, usize)>,
            }

            impl Hh16 {
                pub(crate) const INSTANCE_BYTES: usize = 6 * IN_W * 8;

                pub(crate) fn new(key: &[u8; 32]) -> Self {
                    Self::from_prepared_entropy(entropy::entropy_for_key(key, B::LANES, 6, 7, 2))
                }

                pub(crate) fn from_prepared_entropy(entropy: Entropy) -> Self {
                    Self {
                        stack: core::array::from_fn(|_| [[B::load_one(0); 2]; FANOUT]),
                        stack_lengths: [0; MAX_STACK],
                        buffer: [0; 2048],
                        buffer_len: 0,
                        total_len: 0,
                        entropy,
                        pending: None,
                    }
                }

                #[target_feature(enable = $feature)]
                unsafe fn flush_pending(&mut self) {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        if let Some((ptr, len)) = self.pending.take() {
                            let words = self.entropy.as_slice();
                            let block_count = len / MACRO_BYTES_HH16;
                            dfs_tree_hash16(
                                &mut self.stack,
                                &mut self.stack_lengths,
                                ptr,
                                block_count,
                                words,
                            );
                            let tail_off = block_count * MACRO_BYTES_HH16;
                            let tail_len = len - tail_off;
                            if tail_len > 0 {
                                self.buffer[..tail_len].copy_from_slice(
                                    core::slice::from_raw_parts(ptr.add(tail_off), tail_len),
                                );
                            }
                            self.buffer_len = tail_len;
                        }
                    }
                }

                pub(crate) fn update(&mut self, data: &[u8]) {
                    // SAFETY: Backend `update`/`finalize` requires the matching CPU feature to be enabled.
                    unsafe { self.update_impl(data) }
                }

                #[target_feature(enable = $feature)]
                unsafe fn update_impl(&mut self, data: &[u8]) {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        self.total_len = self.total_len.wrapping_add(data.len() as u64);

                        if self.pending.is_some() {
                            self.flush_pending();
                        }

                        if stack_is_empty(&self.stack_lengths, self.buffer_len) {
                            self.pending = Some((data.as_ptr(), data.len()));
                            return;
                        }

                        let words = self.entropy.as_slice();
                        let mut off = 0usize;
                        if self.buffer_len > 0 {
                            let need = Self::INSTANCE_BYTES - self.buffer_len;
                            let take = need.min(data.len());
                            self.buffer[self.buffer_len..self.buffer_len + take]
                                .copy_from_slice(&data[..take]);
                            self.buffer_len += take;
                            off += take;
                            if self.buffer_len == Self::INSTANCE_BYTES {
                                let mut padded = [0u8; MACRO_BYTES_HH16];
                                padded[..Self::INSTANCE_BYTES]
                                    .copy_from_slice(&self.buffer[..Self::INSTANCE_BYTES]);
                                dfs_tree_hash16(
                                    &mut self.stack,
                                    &mut self.stack_lengths,
                                    padded.as_ptr(),
                                    1,
                                    words,
                                );
                                self.buffer_len = 0;
                            }
                        }

                        let block_count = (data.len() - off) / MACRO_BYTES_HH16;
                        if block_count > 0 {
                            dfs_tree_hash16(
                                &mut self.stack,
                                &mut self.stack_lengths,
                                data.as_ptr().add(off),
                                block_count,
                                words,
                            );
                            off += block_count * MACRO_BYTES_HH16;
                        }

                        if off < data.len() {
                            let rem = data.len() - off;
                            self.buffer[..rem].copy_from_slice(&data[off..]);
                            self.buffer_len = rem;
                        }
                    }
                }

                pub(crate) fn finalize(self) -> [u64; 2] {
                    // SAFETY: Backend `update`/`finalize` requires the matching CPU feature to be enabled.
                    unsafe { self.finalize_impl() }
                }

                #[target_feature(enable = $feature)]
                unsafe fn finalize_impl(self) -> [u64; 2] {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        if let Some((ptr, len)) = self.pending {
                            return hash_v4_hh16(self.entropy.as_slice(), ptr, len);
                        }
                        let fin_base = entropy::finalizer_entropy_base(7, IN_W, FANOUT, 2);
                        greedy_finish2(
                            &self.stack,
                            &self.stack_lengths,
                            self.buffer.as_ptr(),
                            self.buffer_len,
                            self.entropy.as_slice(),
                            fin_base,
                        )
                    }
                }
            }

            /// Fast `Hasher` for `HalftimeHash32` / AVX2×2.
            #[derive(Clone)]
            pub(crate) struct Hh32 {
                stack: [[[B; 4]; FANOUT]; MAX_STACK],
                stack_lengths: [u8; MAX_STACK],
                buffer: [u8; 2048],
                buffer_len: usize,
                total_len: u64,
                entropy: Entropy,
                pending: Option<(*const u8, usize)>,
            }

            impl Hh32 {
                pub(crate) const INSTANCE_BYTES: usize = 7 * IN_W * 8;

                pub(crate) fn new(key: &[u8; 32]) -> Self {
                    Self::from_prepared_entropy(entropy::entropy_for_key(key, B::LANES, 7, 10, 4))
                }

                pub(crate) fn from_prepared_entropy(entropy: Entropy) -> Self {
                    Self {
                        stack: core::array::from_fn(|_| [[B::load_one(0); 4]; FANOUT]),
                        stack_lengths: [0; MAX_STACK],
                        buffer: [0; 2048],
                        buffer_len: 0,
                        total_len: 0,
                        entropy,
                        pending: None,
                    }
                }

                #[target_feature(enable = $feature)]
                unsafe fn flush_pending(&mut self) {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        if let Some((ptr, len)) = self.pending.take() {
                            let words = self.entropy.as_slice();
                            let block_count = len / MACRO_BYTES_HH32;
                            dfs_tree_hash32(
                                &mut self.stack,
                                &mut self.stack_lengths,
                                ptr,
                                block_count,
                                words,
                            );
                            let tail_off = block_count * MACRO_BYTES_HH32;
                            let tail_len = len - tail_off;
                            if tail_len > 0 {
                                self.buffer[..tail_len].copy_from_slice(
                                    core::slice::from_raw_parts(ptr.add(tail_off), tail_len),
                                );
                            }
                            self.buffer_len = tail_len;
                        }
                    }
                }

                pub(crate) fn update(&mut self, data: &[u8]) {
                    // SAFETY: Backend `update`/`finalize` requires the matching CPU feature to be enabled.
                    unsafe { self.update_impl(data) }
                }

                #[target_feature(enable = $feature)]
                unsafe fn update_impl(&mut self, data: &[u8]) {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        self.total_len = self.total_len.wrapping_add(data.len() as u64);

                        if self.pending.is_some() {
                            self.flush_pending();
                        }

                        if stack_is_empty(&self.stack_lengths, self.buffer_len) {
                            self.pending = Some((data.as_ptr(), data.len()));
                            return;
                        }

                        let words = self.entropy.as_slice();
                        let mut off = 0usize;
                        if self.buffer_len > 0 {
                            let need = Self::INSTANCE_BYTES - self.buffer_len;
                            let take = need.min(data.len());
                            self.buffer[self.buffer_len..self.buffer_len + take]
                                .copy_from_slice(&data[..take]);
                            self.buffer_len += take;
                            off += take;
                            if self.buffer_len == Self::INSTANCE_BYTES {
                                let mut padded = [0u8; MACRO_BYTES_HH32];
                                padded[..Self::INSTANCE_BYTES]
                                    .copy_from_slice(&self.buffer[..Self::INSTANCE_BYTES]);
                                dfs_tree_hash32(
                                    &mut self.stack,
                                    &mut self.stack_lengths,
                                    padded.as_ptr(),
                                    1,
                                    words,
                                );
                                self.buffer_len = 0;
                            }
                        }

                        let block_count = (data.len() - off) / MACRO_BYTES_HH32;
                        if block_count > 0 {
                            dfs_tree_hash32(
                                &mut self.stack,
                                &mut self.stack_lengths,
                                data.as_ptr().add(off),
                                block_count,
                                words,
                            );
                            off += block_count * MACRO_BYTES_HH32;
                        }

                        if off < data.len() {
                            let rem = data.len() - off;
                            self.buffer[..rem].copy_from_slice(&data[off..]);
                            self.buffer_len = rem;
                        }
                    }
                }

                pub(crate) fn finalize(self) -> [u64; 4] {
                    // SAFETY: Backend `update`/`finalize` requires the matching CPU feature to be enabled.
                    unsafe { self.finalize_impl() }
                }

                #[target_feature(enable = $feature)]
                unsafe fn finalize_impl(self) -> [u64; 4] {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        if let Some((ptr, len)) = self.pending {
                            return hash_v4_hh32(self.entropy.as_slice(), ptr, len);
                        }
                        let fin_base = entropy::finalizer_entropy_base(10, IN_W, FANOUT, 4);
                        greedy_finish4(
                            &self.stack,
                            &self.stack_lengths,
                            self.buffer.as_ptr(),
                            self.buffer_len,
                            self.entropy.as_slice(),
                            fin_base,
                        )
                    }
                }
            }

            /// Fast `Hasher` for `HalftimeHash40` / AVX2×2.
            #[derive(Clone)]
            pub(crate) struct Hh40 {
                stack: [[[B; 5]; FANOUT]; MAX_STACK],
                stack_lengths: [u8; MAX_STACK],
                buffer: [u8; 2048],
                buffer_len: usize,
                total_len: u64,
                entropy: Entropy,
                pending: Option<(*const u8, usize)>,
            }

            impl Hh40 {
                pub(crate) const INSTANCE_BYTES: usize = 5 * IN_W * 8;

                pub(crate) fn new(key: &[u8; 32]) -> Self {
                    Self::from_prepared_entropy(entropy::entropy_for_key(key, B::LANES, 5, 9, 5))
                }

                pub(crate) fn from_prepared_entropy(entropy: Entropy) -> Self {
                    Self {
                        stack: core::array::from_fn(|_| [[B::load_one(0); 5]; FANOUT]),
                        stack_lengths: [0; MAX_STACK],
                        buffer: [0; 2048],
                        buffer_len: 0,
                        total_len: 0,
                        entropy,
                        pending: None,
                    }
                }

                #[target_feature(enable = $feature)]
                unsafe fn flush_pending(&mut self) {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        if let Some((ptr, len)) = self.pending.take() {
                            let words = self.entropy.as_slice();
                            let block_count = len / MACRO_BYTES_HH40;
                            dfs_tree_hash40(
                                &mut self.stack,
                                &mut self.stack_lengths,
                                ptr,
                                block_count,
                                words,
                            );
                            let tail_off = block_count * MACRO_BYTES_HH40;
                            let tail_len = len - tail_off;
                            if tail_len > 0 {
                                self.buffer[..tail_len].copy_from_slice(
                                    core::slice::from_raw_parts(ptr.add(tail_off), tail_len),
                                );
                            }
                            self.buffer_len = tail_len;
                        }
                    }
                }

                pub(crate) fn update(&mut self, data: &[u8]) {
                    // SAFETY: Backend `update`/`finalize` requires the matching CPU feature to be enabled.
                    unsafe { self.update_impl(data) }
                }

                #[target_feature(enable = $feature)]
                unsafe fn update_impl(&mut self, data: &[u8]) {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        self.total_len = self.total_len.wrapping_add(data.len() as u64);

                        if self.pending.is_some() {
                            self.flush_pending();
                        }

                        if stack_is_empty(&self.stack_lengths, self.buffer_len) {
                            self.pending = Some((data.as_ptr(), data.len()));
                            return;
                        }

                        let words = self.entropy.as_slice();
                        let mut off = 0usize;
                        if self.buffer_len > 0 {
                            let need = Self::INSTANCE_BYTES - self.buffer_len;
                            let take = need.min(data.len());
                            self.buffer[self.buffer_len..self.buffer_len + take]
                                .copy_from_slice(&data[..take]);
                            self.buffer_len += take;
                            off += take;
                            if self.buffer_len == Self::INSTANCE_BYTES {
                                let mut padded = [0u8; MACRO_BYTES_HH40];
                                padded[..Self::INSTANCE_BYTES]
                                    .copy_from_slice(&self.buffer[..Self::INSTANCE_BYTES]);
                                dfs_tree_hash40(
                                    &mut self.stack,
                                    &mut self.stack_lengths,
                                    padded.as_ptr(),
                                    1,
                                    words,
                                );
                                self.buffer_len = 0;
                            }
                        }

                        let block_count = (data.len() - off) / MACRO_BYTES_HH40;
                        if block_count > 0 {
                            dfs_tree_hash40(
                                &mut self.stack,
                                &mut self.stack_lengths,
                                data.as_ptr().add(off),
                                block_count,
                                words,
                            );
                            off += block_count * MACRO_BYTES_HH40;
                        }

                        if off < data.len() {
                            let rem = data.len() - off;
                            self.buffer[..rem].copy_from_slice(&data[off..]);
                            self.buffer_len = rem;
                        }
                    }
                }

                pub(crate) fn finalize(self) -> [u64; 5] {
                    // SAFETY: Backend `update`/`finalize` requires the matching CPU feature to be enabled.
                    unsafe { self.finalize_impl() }
                }

                #[target_feature(enable = $feature)]
                unsafe fn finalize_impl(self) -> [u64; 5] {
                    // SAFETY: Body uses raw pointers and `#[target_feature]` helpers for this fused backend.
                    unsafe {
                        if let Some((ptr, len)) = self.pending {
                            return hash_v4_hh40(self.entropy.as_slice(), ptr, len);
                        }
                        let fin_base = entropy::finalizer_entropy_base(9, IN_W, FANOUT, 5);
                        greedy_finish5(
                            &self.stack,
                            &self.stack_lengths,
                            self.buffer.as_ptr(),
                            self.buffer_len,
                            self.entropy.as_slice(),
                            fin_base,
                        )
                    }
                }
            }
        }
    };
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
define_fused_backend!(avx512, crate::block::Avx512Block, "avx512f");

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
define_fused_backend!(avx2, crate::block::Avx2Repeat2Block, "avx2");

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
define_fused_backend!(
    sse2,
    crate::block::RepeatBlock<crate::block::Sse2Block, 4>,
    "sse2"
);

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
define_fused_backend!(
    neon,
    crate::block::RepeatBlock<crate::block::NeonBlock, 4>,
    "neon"
);
