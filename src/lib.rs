#![no_std]
#![doc = include_str!("../README.md")]

extern crate alloc;

pub use universal_hash;

mod backend;
mod block;
mod ehc_badger;
mod entropy;
mod hash;
mod key;
mod nh;
mod variant;
mod word;

pub use entropy::{ENTROPY_BLOCK_LANES, EntropyTooShort, entropy_words_needed};
pub use hash::{
    HalftimeHash16, HalftimeHash24, HalftimeHash32, HalftimeHash40, KEY_SIZE, Tag, Tag16, Tag24,
    Tag32, Tag40,
};
pub use key::{Key, Key32, Key64, Key128, MasterKey};
pub use variant::{Hh16, Hh24, Hh32, Hh40};
