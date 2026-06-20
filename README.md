# HalftimeHash

[![Crates.io](https://img.shields.io/crates/v/halftime.svg)](https://crates.io/crates/halftime)
[![Documentation](https://docs.rs/halftime/badge.svg)](https://docs.rs/halftime)
[![License](https://img.shields.io/crates/l/halftime.svg)](https://crates.io/crates/halftime)
[![Coverage Status](https://coveralls.io/repos/github/ShaJaPas/halftime-rs/badge.svg)](https://coveralls.io/github/ShaJaPas/halftime-rs)

Almost-universal string hashing for long inputs, from the [HalftimeHash paper (2020)](https://arxiv.org/abs/2104.08865).

This crate implements the four output widths from Section 4 of the paper as a `#![no_std]` Rust library. Each variant implements RustCrypto's [`UniversalHash`](https://docs.rs/universal-hash) trait and is optimized for throughput on large messages with SIMD (AVX-512, AVX2, SSE2 on x86-64; NEON on AArch64).

## Features

- **Four tag widths** — 16, 24, 32, and 40 bytes with different almost-universality margins.
- **Two keying modes** — 32-byte master key with NH KDF (convenient), or external input entropy of arbitrary length (C++ reference style).
- **Master keys** — [`Key32`], [`Key64`], [`Key128`] (byte count); any key works with any variant.
- **`no_std`** — works without the standard library; requires **`alloc`** because tree-entropy expansion grows a dynamic word buffer (`Vec`/`Arc`) as input length increases. Memory use scales with **O(log n)** in the input size (tree depth), not with the full message.
- **Runtime CPU dispatch** on x86-64 (AVX-512 → AVX2 → SSE2 → scalar fallback).
- **RustCrypto-compatible API** — `KeyInit`, `UniversalHash`, one-shot `digest`.

## When to use HalftimeHash

HalftimeHash is a **universal hash**, not a standalone MAC. It maps `(key, message) → tag` with strong almost-universality guarantees for long strings, but provides **no confidentiality, no nonce handling, and no key commitment** on its own.

**Good fits:**

- **Long-message authentication** where throughput dominates — bulk data, large records, log streams, file chunks, database pages.
- **Custom authenticated-encryption or MAC-then-encrypt designs** where you control the full protocol and combine a universal hash with a cipher and a nonce (as in the paper's AEAD construction).
- **Tag-size tradeoffs** — pick 16/24/32/40 bytes depending on your collision budget; wider tags give more security margin at some CPU cost.
- **Experimental or research systems** exploring high-throughput almost-universal hashing.

**Poor fits — use something else:**

- **Standard AEAD** (TLS, Noise, `ChaCha20-Poly1305`, AES-GCM) — Poly1305 and GHASH are the interoperable choices.
- **Short messages** (≤ a few KiB) — fixed setup cost favors Poly1305 and other lightweight MACs (see benchmarks below).
- **Standalone message authentication** without a surrounding AEAD or encrypt-then-MAC scheme — a universal hash alone is not a complete integrity mechanism.
- **Side-channel–sensitive contexts** — this implementation prioritizes SIMD throughput; it is **not** designed as a constant-time MAC like Poly1305.

## Comparison with Poly1305

[Poly1305](https://docs.rs/poly1305) is the de facto universal hash in production AEAD (notably `ChaCha20-Poly1305`). It produces a **16-byte tag**, has a **32-byte key**, is **fast on short inputs**, and is **widely audited**. HalftimeHash24 produces a **24-byte tag** and is designed to **win on long strings**.

Benchmarks below were measured with `cargo bench --bench halftime` on a **single core** of x86-64 (release, LTO). Both hashes receive identical random-looking input; HalftimeHash24 uses `digest` (one-shot).

| Input size | HalftimeHash24 | Poly1305 | Winner |
|------------|----------------|----------|--------|
| 64 B | 16 MiB/s | 47 MiB/s | Poly1305 (~3×) |
| 256 B | 66 MiB/s | 184 MiB/s | Poly1305 (~3×) |
| 1 KiB | 260 MiB/s | 657 MiB/s | Poly1305 (~2.5×) |
| 4 KiB | 1.0 GiB/s | 1.7 GiB/s | Poly1305 (~1.7×) |
| 64 KiB | 11.8 GiB/s | 3.9 GiB/s | **HalftimeHash24 (~3×)** |
| 256 KiB | 23.5 GiB/s | 4.1 GiB/s | **HalftimeHash24 (~5.7×)** |
| 1 MiB | 31.2 GiB/s | 4.3 GiB/s | **HalftimeHash24 (~7.3×)** |

**Takeaway:** Poly1305 stays ahead while the message is small enough that setup and per-block overhead matter. Beyond roughly **tens of KiB**, HalftimeHash's tree-based design amortizes its cost and pulls ahead; at **1 MiB** it is about **7× faster** than Poly1305 on the same hardware, while emitting a wider tag.

HalftimeHash16 is even faster on very long inputs (~35 GiB/s at 1 MiB in the same benchmark suite) with a 16-byte tag, at a lower almost-universality margin than HH24.

## Variants

| Type | Tag size | Paper parameters | Typical use |
|------|----------|------------------|-------------|
| `HalftimeHash16` | 16 bytes | distance-2 code | Maximum speed, shortest tag |
| `HalftimeHash24` | 24 bytes | distance-3 code | Default balance (see `Tag` alias) |
| `HalftimeHash32` | 32 bytes | distance-4 code | Stronger margin, still fast |
| `HalftimeHash40` | 40 bytes | distance-5 code | Widest tag, strongest margin |

All variants accept the same master key types ([`Key32`], [`Key64`], [`Key128`]). Tag types are [`Tag16`] … [`Tag40`].

## Keys and entropy

HalftimeHash needs a large stream of **input entropy** (random `u64` words) that grows with message length — see Section 4.3 of the paper. This crate supports two ways to supply it:

### Master key + NH KDF (default)

Pass a fixed master key ([`Key32`], [`Key64`], or [`Key128`]). The crate derives 32-byte seed material (NH fold for longer keys) and expands entropy on demand with NH — convenient when you only have a short secret key.

```rust
use halftime::{HalftimeHash24, Key32};

let key = Key32::from([0x42u8; 32]); // CSPRNG or KDF in production
let tag = HalftimeHash24::digest_master_key(&key, b"message");
```

[`KeyInit`] / `HalftimeHash24::new` accept a 32-byte RustCrypto key and use the same NH expansion.

### External entropy (C++ reference style)

Supply a pre-generated `&[u64]` buffer, as in the C++ `const uint64_t* entropy` API. Size it with [`HalftimeHash24::entropy_words_needed`] (or the free function [`entropy_words_needed`]):

```rust
use halftime::HalftimeHash24;

let input = b"message to authenticate";
let need = HalftimeHash24::entropy_words_needed(input.len());
// Fill `entropy` from a CSPRNG in production (one u64 per word).
let entropy: Vec<u64> = vec![0; need];

let tag = HalftimeHash24::digest_with_entropy(&entropy, input).expect("entropy too short");
```

For incremental hashing with external entropy, call [`HalftimeHash24::with_entropy_for`] with your maximum expected message length.

**Note:** [`HalftimeHash32`] is the variant with a 32-byte **tag**; [`Key32`] is a 32-byte **master key** — different concepts.


## Usage

### One-shot digest

```rust
use halftime::{HalftimeHash24, Key32};

// In production, obtain these 32 bytes from a CSPRNG (e.g. `getrandom`) or a KDF (e.g. HKDF).
let key_bytes = [0x42u8; 32];
let key = Key32::from(key_bytes);

let tag = HalftimeHash24::digest_master_key(&key, b"message to authenticate");
assert_eq!(tag.as_slice().len(), HalftimeHash24::TAG_BYTES);
```

### Incremental hashing

```rust
use halftime::{HalftimeHash24, Key32, universal_hash::UniversalHash};

// In production, obtain these 32 bytes from a CSPRNG (e.g. `getrandom`) or a KDF (e.g. HKDF).
let key_bytes = [0x42u8; 32];
let key = Key32::from(key_bytes);

let mut mac = HalftimeHash24::from_master_key(&key);
mac.update_padded(b"first chunk");
mac.update_padded(b"second chunk");
let tag = mac.finalize();
```

### Choosing a variant

```rust
use halftime::{HalftimeHash16, HalftimeHash40, Key32};

// Same key type for every variant — the number is key bytes, not tag width.
let key = Key32::from([0x42u8; 32]);
let tag16 = HalftimeHash16::digest_master_key(&key, b"large blob"); // 16-byte tag, fastest
let tag40 = HalftimeHash40::digest_master_key(&key, b"large blob"); // 40-byte tag, strongest margin
```

## Performance

Run the full suite:

```bash
cargo bench --bench halftime
```

Useful filters:

```bash
# Head-to-head vs Poly1305
cargo bench --bench halftime -- compare/HalftimeHash24_vs_Poly1305

# All variants at 256 KiB (paper comparison point)
cargo bench --bench halftime -- long_input/256KiB

# Single variant, long input only
cargo bench --bench halftime -- "HalftimeHash16/hash_only/1048576"
```


### Platform notes

| Platform | Backend |
|----------|---------|
| x86-64 | AVX-512 if available, else AVX2, else SSE2, else scalar |
| AArch64 | NEON |
| Other | Scalar (`RepeatBlock` over `ScalarBlock`) |

Force the portable scalar backend for testing:

```bash
RUSTFLAGS='--cfg halftime_backend="soft"' cargo build
```

## Security notes

- HalftimeHash is **almost-universal**, not collision-resistant or a PRF. Treat tags as authentication values inside a proper AEAD or MAC scheme, not as general-purpose hashes.
- **Generate keys from a CSPRNG or KDF** — never use predictable placeholder bytes in production.
- **Never reuse a `(key, nonce) pair`** across different messages in an AEAD construction (same rule as Poly1305 in `ChaCha20-Poly1305`).
- Wider variants (HH32, HH40) increase the almost-universality margin at the cost of CPU time; match tag width to your threat model.
- This crate has **not** received the same level of public audit as Poly1305. Evaluate carefully before production deployment.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
