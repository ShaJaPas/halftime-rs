//! Scalar and SIMD backends via [`ehc_badger::Hasher`].

use crate::variant::{Hh16, Hh24, Hh32, Hh40};

use super::state::{HasherFromEntropy, HasherFromKey, State};

#[cfg(not(all(target_arch = "aarch64", not(halftime_backend = "soft"))))]
use super::state::PlainFeed;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
use super::state::{Avx2Feed, Avx512Feed, Sse2Feed};

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
use super::state::NeonFeed;

#[cfg(all(not(halftime_backend = "soft"), not(target_arch = "aarch64")))]
use crate::ehc_badger::{Repeat8Hasher16, Repeat8Hasher24, Repeat8Hasher32, Repeat8Hasher40};

#[cfg(halftime_backend = "soft")]
use crate::ehc_badger::{Hasher16, Hasher24, Hasher32, Hasher40};

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
use crate::ehc_badger::{
    Avx2Repeat8Hasher16, Avx2Repeat8Hasher24, Avx2Repeat8Hasher32, Avx2Repeat8Hasher40,
    Avx512Hasher16, Avx512Hasher24, Avx512Hasher32, Avx512Hasher40, Sse2Repeat8Hasher16,
    Sse2Repeat8Hasher24, Sse2Repeat8Hasher32, Sse2Repeat8Hasher40,
};

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
use crate::ehc_badger::{Neon8Hasher16, Neon8Hasher24, Neon8Hasher32, Neon8Hasher40};

macro_rules! impl_hasher_from_key {
    ($($t:ty),* $(,)?) => {
        $(
            impl HasherFromKey for $t {
                #[inline]
                fn from_key(key: &[u8; 32]) -> Self {
                    Self::new(key)
                }
            }

            impl HasherFromEntropy for $t {
                #[inline]
                fn from_entropy(entropy: crate::entropy::Entropy) -> Self {
                    Self::from_prepared_entropy(entropy)
                }
            }
        )*
    };
}

#[cfg(halftime_backend = "soft")]
impl_hasher_from_key!(Hasher16, Hasher24, Hasher32, Hasher40);

#[cfg(all(not(halftime_backend = "soft"), not(target_arch = "aarch64")))]
impl_hasher_from_key!(
    Repeat8Hasher16,
    Repeat8Hasher24,
    Repeat8Hasher32,
    Repeat8Hasher40
);

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
impl_hasher_from_key!(
    Avx2Repeat8Hasher16,
    Avx2Repeat8Hasher24,
    Avx2Repeat8Hasher32,
    Avx2Repeat8Hasher40,
    Sse2Repeat8Hasher16,
    Sse2Repeat8Hasher24,
    Sse2Repeat8Hasher32,
    Sse2Repeat8Hasher40,
    Avx512Hasher16,
    Avx512Hasher24,
    Avx512Hasher32,
    Avx512Hasher40,
);

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
impl_hasher_from_key!(Neon8Hasher16, Neon8Hasher24, Neon8Hasher32, Neon8Hasher40);

#[cfg(halftime_backend = "soft")]
pub(crate) type State16 = State<Hh16, Hasher16, PlainFeed>;
#[cfg(halftime_backend = "soft")]
pub(crate) type State24 = State<Hh24, Hasher24, PlainFeed>;
#[cfg(halftime_backend = "soft")]
pub(crate) type State32 = State<Hh32, Hasher32, PlainFeed>;
#[cfg(halftime_backend = "soft")]
pub(crate) type State40 = State<Hh40, Hasher40, PlainFeed>;

#[cfg(all(not(halftime_backend = "soft"), not(target_arch = "aarch64")))]
pub(crate) type State16 = State<Hh16, Repeat8Hasher16, PlainFeed>;
#[cfg(all(not(halftime_backend = "soft"), not(target_arch = "aarch64")))]
pub(crate) type State24 = State<Hh24, Repeat8Hasher24, PlainFeed>;
#[cfg(all(not(halftime_backend = "soft"), not(target_arch = "aarch64")))]
pub(crate) type State32 = State<Hh32, Repeat8Hasher32, PlainFeed>;
#[cfg(all(not(halftime_backend = "soft"), not(target_arch = "aarch64")))]
pub(crate) type State40 = State<Hh40, Repeat8Hasher40, PlainFeed>;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx2Repeat8State16 = State<Hh16, Avx2Repeat8Hasher16, Avx2Feed>;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx2Repeat8State24 = State<Hh24, Avx2Repeat8Hasher24, Avx2Feed>;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx2Repeat8State32 = State<Hh32, Avx2Repeat8Hasher32, Avx2Feed>;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx2Repeat8State40 = State<Hh40, Avx2Repeat8Hasher40, Avx2Feed>;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Sse2Repeat8State16 = State<Hh16, Sse2Repeat8Hasher16, Sse2Feed>;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Sse2Repeat8State24 = State<Hh24, Sse2Repeat8Hasher24, Sse2Feed>;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Sse2Repeat8State32 = State<Hh32, Sse2Repeat8Hasher32, Sse2Feed>;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Sse2Repeat8State40 = State<Hh40, Sse2Repeat8Hasher40, Sse2Feed>;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx512State16 = State<Hh16, Avx512Hasher16, Avx512Feed>;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx512State24 = State<Hh24, Avx512Hasher24, Avx512Feed>;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx512State32 = State<Hh32, Avx512Hasher32, Avx512Feed>;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(halftime_backend = "soft")
))]
pub(crate) type Avx512State40 = State<Hh40, Avx512Hasher40, Avx512Feed>;

#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) type NeonState16 = State<Hh16, Neon8Hasher16, NeonFeed>;
#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) type NeonState24 = State<Hh24, Neon8Hasher24, NeonFeed>;
#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) type NeonState32 = State<Hh32, Neon8Hasher32, NeonFeed>;
#[cfg(all(target_arch = "aarch64", not(halftime_backend = "soft")))]
pub(crate) type NeonState40 = State<Hh40, Neon8Hasher40, NeonFeed>;
