//! Throughput benchmarks: HalftimeHash vs Poly1305 on identical inputs.
//!
//! Reports bytes per second via Criterion (`cargo bench --bench halftime`).

use core::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use halftime::{
    HalftimeHash16, HalftimeHash24, HalftimeHash32, HalftimeHash40, Key32,
    universal_hash::{KeyInit, UniversalHash},
};
use poly1305::{Key as Poly1305Key, Poly1305};

/// Input lengths emphasized in the paper (short → long, including 250 KiB and 256 KiB).
const INPUT_SIZES: &[usize] = &[
    64,
    256,
    1 << 10,
    1 << 12,
    1 << 16,
    250 * 1024,
    256 * 1024,
    1 << 20,
];

#[inline]
fn poly1305_digest(key: &Poly1305Key, data: &[u8]) -> poly1305::Tag {
    let mut mac = Poly1305::new(key);
    mac.update_padded(data);
    mac.finalize()
}

macro_rules! bench_halftime_variant {
    ($c:expr, $name:ident) => {{
        let key = universal_hash::Key::<$name>::default();

        {
            let mut group = $c.benchmark_group(concat!(stringify!($name), "/digest"));
            for &len in INPUT_SIZES {
                let data = vec![0xA5u8; len];
                group.throughput(Throughput::Bytes(len as u64));
                group.bench_with_input(BenchmarkId::from_parameter(len), &data, |b, data| {
                    b.iter(|| black_box($name::digest(black_box(&key), black_box(data))));
                });
            }
            group.finish();
        }

        {
            let mut group = $c.benchmark_group(concat!(stringify!($name), "/incremental"));
            for &len in INPUT_SIZES {
                let data = vec![0xA5u8; len];
                group.throughput(Throughput::Bytes(len as u64));
                group.bench_with_input(BenchmarkId::from_parameter(len), &data, |b, data| {
                    b.iter(|| {
                        let mut h = $name::new(black_box(&key));
                        h.update_padded(black_box(data));
                        black_box(h.finalize())
                    });
                });
            }
            group.finish();
        }

        {
            let mut group = $c.benchmark_group(concat!(stringify!($name), "/hash_only"));
            for &len in INPUT_SIZES {
                let data = vec![0xA5u8; len];
                group.throughput(Throughput::Bytes(len as u64));
                group.bench_with_input(BenchmarkId::from_parameter(len), &data, |b, data| {
                    b.iter_batched(
                        || $name::new(black_box(&key)),
                        |mut h| {
                            h.update_padded(black_box(data));
                            black_box(h.finalize())
                        },
                        BatchSize::LargeInput,
                    );
                });
            }
            group.finish();
        }
    }};
}

/// Head-to-head on the same buffer (expect Poly1305 ≈2× HalftimeHash24 on long input).
fn compare_halftime24_poly1305(c: &mut Criterion) {
    let hh_key = universal_hash::Key::<HalftimeHash24>::default();
    let p_key = Poly1305Key::default();
    let mut group = c.benchmark_group("compare/HalftimeHash24_vs_Poly1305");

    for &len in INPUT_SIZES {
        let data = vec![0xA5u8; len];
        group.throughput(Throughput::Bytes(len as u64));

        group.bench_with_input(BenchmarkId::new("HalftimeHash24", len), &data, |b, data| {
            b.iter(|| black_box(HalftimeHash24::digest(black_box(&hh_key), black_box(data))));
        });
        group.bench_with_input(BenchmarkId::new("Poly1305", len), &data, |b, data| {
            b.iter(|| black_box(poly1305_digest(black_box(&p_key), black_box(data))));
        });
    }

    group.finish();
}

fn halftime16(c: &mut Criterion) {
    bench_halftime_variant!(c, HalftimeHash16);
}

fn halftime24(c: &mut Criterion) {
    bench_halftime_variant!(c, HalftimeHash24);
}

fn halftime32(c: &mut Criterion) {
    bench_halftime_variant!(c, HalftimeHash32);
}

fn halftime40(c: &mut Criterion) {
    bench_halftime_variant!(c, HalftimeHash40);
}

/// Paper Figure `frontier` / `vs-cl`: compare output widths at 256 KiB.
fn long_input_variants(c: &mut Criterion) {
    let len = 256 * 1024;
    let data = vec![0xA5u8; len];
    let key = Key32::from([0xA5u8; 32]);
    let p_key = Poly1305Key::default();

    let mut group = c.benchmark_group("long_input/256KiB");
    group.throughput(Throughput::Bytes(len as u64));

    group.bench_function("HalftimeHash16", |b| {
        b.iter(|| {
            black_box(HalftimeHash16::digest_master_key(
                black_box(&key),
                black_box(&data),
            ))
        });
    });
    group.bench_function("HalftimeHash24", |b| {
        b.iter(|| {
            black_box(HalftimeHash24::digest_master_key(
                black_box(&key),
                black_box(&data),
            ))
        });
    });
    group.bench_function("HalftimeHash32", |b| {
        b.iter(|| {
            black_box(HalftimeHash32::digest_master_key(
                black_box(&key),
                black_box(&data),
            ))
        });
    });
    group.bench_function("HalftimeHash40", |b| {
        b.iter(|| {
            black_box(HalftimeHash40::digest_master_key(
                black_box(&key),
                black_box(&data),
            ))
        });
    });
    group.bench_function("Poly1305", |b| {
        b.iter(|| black_box(poly1305_digest(black_box(&p_key), black_box(&data))));
    });

    group.finish();
}

criterion_group!(
    benches,
    halftime16,
    halftime24,
    halftime32,
    halftime40,
    compare_halftime24_poly1305,
    long_input_variants
);
criterion_main!(benches);
