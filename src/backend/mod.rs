//! Backend implementations (scalar with optional CPU dispatch).

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) mod autodetect;

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) mod neon;

pub(crate) mod soft;

pub(crate) mod state;

pub(crate) mod feed;

pub(crate) trait Absorb {
    fn absorb(&mut self, data: &[u8]);
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) use autodetect::{State16, State24, State32, State40};

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) use neon::{State16, State24, State32, State40};

#[cfg(not(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(halftime_backend = "soft")
    ),
    all(target_arch = "aarch64", not(halftime_backend = "soft"))
)))]
pub(crate) use soft::{State16, State24, State32, State40};
