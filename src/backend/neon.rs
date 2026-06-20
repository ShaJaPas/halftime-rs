//! AArch64 backend (`RepeatWrapper` over NEON, `b = 8`).

pub(crate) use super::soft::{
    NeonState16 as State16, NeonState24 as State24, NeonState32 as State32, NeonState40 as State40,
};
