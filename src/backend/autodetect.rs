//! Runtime CPU feature detection (x86): AVX-512 > AVX2×2 > SSE2×4 > scalar repeat.

use universal_hash::{UhfClosure, UniversalHash, common::BlockSizeUser};

use crate::backend::{Absorb, soft};
use crate::entropy::Entropy;
use crate::variant::HalftimeVariant;

cpufeatures::new!(avx2_token, "avx2");
cpufeatures::new!(avx512f_token, "avx512f");
cpufeatures::new!(sse2_token, "sse2");

macro_rules! define_autodetect {
    ($mod_name:ident, $v:ty, $avx512:ty, $avx2r8:ty, $sse2r8:ty, $repeat8:ty) => {
        pub(crate) mod $mod_name {
            use super::*;

            pub(crate) enum Inner {
                Avx512($avx512),
                Avx2Repeat8($avx2r8),
                Sse2Repeat8($sse2r8),
                Repeat8($repeat8),
            }

            pub(crate) struct State {
                inner: Inner,
            }

            impl Clone for State {
                fn clone(&self) -> Self {
                    Self {
                        inner: match &self.inner {
                            Inner::Avx512(s) => Inner::Avx512(s.clone()),
                            Inner::Avx2Repeat8(s) => Inner::Avx2Repeat8(s.clone()),
                            Inner::Sse2Repeat8(s) => Inner::Sse2Repeat8(s.clone()),
                            Inner::Repeat8(s) => Inner::Repeat8(s.clone()),
                        },
                    }
                }
            }

            impl BlockSizeUser for State {
                type BlockSize = <$v as HalftimeVariant>::TagSize;
            }

            impl State {
                pub(crate) fn new(key: &[u8; 32]) -> Self {
                    let (_avx512_token, has_avx512) = avx512f_token::init_get();
                    let (_avx2_token, has_avx2) = avx2_token::init_get();
                    let (_sse2_token, has_sse2) = sse2_token::init_get();
                    let inner = if has_avx512 {
                        Inner::Avx512(<$avx512>::new(key))
                    } else if has_avx2 {
                        Inner::Avx2Repeat8(<$avx2r8>::new(key))
                    } else if has_sse2 {
                        Inner::Sse2Repeat8(<$sse2r8>::new(key))
                    } else {
                        Inner::Repeat8(<$repeat8>::new(key))
                    };
                    Self { inner }
                }

                pub(crate) fn from_entropy(entropy: Entropy) -> Self {
                    let (_avx512_token, has_avx512) = avx512f_token::init_get();
                    let (_avx2_token, has_avx2) = avx2_token::init_get();
                    let (_sse2_token, has_sse2) = sse2_token::init_get();
                    let inner = if has_avx512 {
                        Inner::Avx512(<$avx512>::from_entropy(entropy))
                    } else if has_avx2 {
                        Inner::Avx2Repeat8(<$avx2r8>::from_entropy(entropy.clone()))
                    } else if has_sse2 {
                        Inner::Sse2Repeat8(<$sse2r8>::from_entropy(entropy.clone()))
                    } else {
                        Inner::Repeat8(<$repeat8>::from_entropy(entropy))
                    };
                    Self { inner }
                }
            }

            impl Absorb for State {
                fn absorb(&mut self, data: &[u8]) {
                    match &mut self.inner {
                        Inner::Avx512(s) => s.absorb(data),
                        Inner::Avx2Repeat8(s) => s.absorb(data),
                        Inner::Sse2Repeat8(s) => s.absorb(data),
                        Inner::Repeat8(s) => s.absorb(data),
                    }
                }
            }

            impl UniversalHash for State {
                fn update_with_backend(&mut self, f: impl UhfClosure<BlockSize = Self::BlockSize>) {
                    match &mut self.inner {
                        Inner::Avx512(s) => f.call(s),
                        Inner::Avx2Repeat8(s) => f.call(s),
                        Inner::Sse2Repeat8(s) => f.call(s),
                        Inner::Repeat8(s) => f.call(s),
                    }
                }

                fn finalize(self) -> universal_hash::array::Array<u8, Self::BlockSize> {
                    match self.inner {
                        Inner::Avx512(s) => s.finalize(),
                        Inner::Avx2Repeat8(s) => s.finalize(),
                        Inner::Sse2Repeat8(s) => s.finalize(),
                        Inner::Repeat8(s) => s.finalize(),
                    }
                }
            }
        }
    };
}

macro_rules! autodetect_variants {
    ($($mod:ident, $v:ty, $avx512:ty, $avx2:ty, $sse2:ty, $r8:ty;)*) => {
        $(define_autodetect!($mod, $v, $avx512, $avx2, $sse2, $r8);)*
    };
}

use crate::variant::{Hh16, Hh24, Hh32, Hh40};

autodetect_variants! {
    ad16, Hh16, soft::Avx512State16, soft::Avx2Repeat8State16, soft::Sse2Repeat8State16, soft::State16;
    ad24, Hh24, soft::Avx512State24, soft::Avx2Repeat8State24, soft::Sse2Repeat8State24, soft::State24;
    ad32, Hh32, soft::Avx512State32, soft::Avx2Repeat8State32, soft::Sse2Repeat8State32, soft::State32;
    ad40, Hh40, soft::Avx512State40, soft::Avx2Repeat8State40, soft::Sse2Repeat8State40, soft::State40;
}

pub(crate) use ad16::State as State16;
pub(crate) use ad24::State as State24;
pub(crate) use ad32::State as State32;
pub(crate) use ad40::State as State40;
