//! Monolithic SIMD pipelines (V4, `b = 8`) with runtime `target_feature` dispatch.

mod pipeline;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) use pipeline::{avx2, avx512, sse2};

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) use pipeline::neon;
