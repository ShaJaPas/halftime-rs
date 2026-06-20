//! Generic Gabrielyan erasure encode (EHC step 1) over [`Block`].

use crate::block::Block;

type Row3<B> = [B; 3];

#[inline(always)]
fn xor3<B: Block>(a: Row3<B>, b: Row3<B>) -> Row3<B> {
    [a[0].xor(b[0]), a[1].xor(b[1]), a[2].xor(b[2])]
}

#[inline(always)]
fn m1<B: Block>(r: Row3<B>) -> Row3<B> {
    [r[1], r[2], r[0].xor(r[1])]
}

#[inline(always)]
fn m2<B: Block>(r: Row3<B>) -> Row3<B> {
    [r[0].xor(r[1]), r[1].xor(r[2]), r[0].xor(r[1]).xor(r[2])]
}

#[inline(always)]
fn m3<B: Block>(r: Row3<B>) -> Row3<B> {
    [r[2], r[0].xor(r[1]), r[1].xor(r[2])]
}

#[inline(always)]
fn m4<B: Block>(r: Row3<B>) -> Row3<B> {
    [r[0].xor(r[2]), r[0], r[1]]
}

#[inline(always)]
fn m5<B: Block>(r: Row3<B>) -> Row3<B> {
    [r[1].xor(r[2]), r[0].xor(r[1]).xor(r[2]), r[0].xor(r[2])]
}

#[inline(always)]
fn m6<B: Block>(r: Row3<B>) -> Row3<B> {
    [r[0].xor(r[1]).xor(r[2]), r[0].xor(r[2]), r[0]]
}

/// Distance-2 XOR parity (`HalftimeHash16`).
#[inline(always)]
pub(crate) fn encode2<B: Block>(io: &mut [Row3<B>; 7]) {
    let mut p = io[0];
    p = xor3(p, io[1]);
    p = xor3(p, io[2]);
    p = xor3(p, io[3]);
    p = xor3(p, io[4]);
    p = xor3(p, io[5]);
    io[6] = p;
}

/// Distance-3 Gabrielyan (9, 7) code (`HalftimeHash24`).
#[inline(always)]
pub(crate) fn encode3<B: Block>(io: &mut [Row3<B>; 9]) {
    let mut p7 = io[0];
    p7 = xor3(p7, io[1]);
    p7 = xor3(p7, io[2]);
    p7 = xor3(p7, io[3]);
    p7 = xor3(p7, io[4]);
    p7 = xor3(p7, io[5]);
    p7 = xor3(p7, io[6]);
    io[7] = p7;

    let mut p8 = io[0];
    p8 = xor3(p8, m1(io[1]));
    p8 = xor3(p8, m2(io[2]));
    p8 = xor3(p8, m3(io[3]));
    p8 = xor3(p8, m4(io[4]));
    p8 = xor3(p8, m5(io[5]));
    p8 = xor3(p8, m6(io[6]));
    io[8] = p8;
}

/// Distance-4 Gabrielyan (10, 7) code (`HalftimeHash32`).
#[inline(always)]
pub(crate) fn encode4<B: Block>(io: &mut [Row3<B>; 10]) {
    let mut p7 = io[0];
    p7 = xor3(p7, io[1]);
    p7 = xor3(p7, io[2]);
    p7 = xor3(p7, io[3]);
    p7 = xor3(p7, io[4]);
    p7 = xor3(p7, io[5]);
    p7 = xor3(p7, io[6]);
    io[7] = p7;

    let mut p8 = io[0];
    p8 = xor3(p8, m1(io[1]));
    p8 = xor3(p8, m2(io[2]));
    p8 = xor3(p8, m3(io[3]));
    p8 = xor3(p8, m4(io[4]));
    p8 = xor3(p8, m5(io[5]));
    p8 = xor3(p8, m6(io[6]));
    io[8] = p8;

    let mut p9 = io[0];
    p9 = xor3(p9, m2(io[1]));
    p9 = xor3(p9, m4(io[2]));
    p9 = xor3(p9, m1(io[3]));
    p9 = xor3(p9, m5(io[4]));
    p9 = xor3(p9, m6(io[5]));
    p9 = xor3(p9, m3(io[6]));
    io[9] = p9;
}

/// Distance-5 Gabrielyan (9, 5) code (`HalftimeHash40`).
#[inline(always)]
pub(crate) fn encode5<B: Block>(io: &mut [Row3<B>; 9]) {
    let mut p5 = io[0];
    p5 = xor3(p5, io[1]);
    p5 = xor3(p5, io[2]);
    p5 = xor3(p5, io[3]);
    p5 = xor3(p5, io[4]);
    io[5] = p5;

    let mut p6 = io[0];
    p6 = xor3(p6, m1(io[1]));
    p6 = xor3(p6, m2(io[2]));
    p6 = xor3(p6, m3(io[3]));
    p6 = xor3(p6, m4(io[4]));
    io[6] = p6;

    let p_init = m1(io[0]);

    let mut p7 = p_init;
    p7 = xor3(p7, m6(io[1]));
    p7 = xor3(p7, m2(io[2]));
    p7 = xor3(p7, io[3]);
    p7 = xor3(p7, m3(io[4]));
    io[7] = p7;

    let mut p8 = p_init;
    p8 = xor3(p8, io[1]);
    p8 = xor3(p8, m4(io[2]));
    p8 = xor3(p8, m5(io[3]));
    p8 = xor3(p8, m2(io[4]));
    io[8] = p8;
}
