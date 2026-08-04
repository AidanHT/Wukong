//! A fast, dependency-free FxHash hasher shared across the front-end.
//!
//! `std`'s default `HashMap`/`HashSet` use SipHash: DoS-resistant, but slow — a fixed per-key cost
//! that dominates when the keys are tiny and the maps are hit millions of times. The front-end keys
//! its maps on interned `Symbol`s, small integers, and short identifier strings drawn from the
//! program being compiled, never from untrusted network input, so SipHash's collision resistance
//! buys nothing. This is the "FxHash" — rotate, xor, multiply — that `rustc` and Firefox use
//! internally for exactly this reason. It is dependency-free, so `wukong_span` stays a zero-dep leaf
//! crate and `cargo test` stays toolchain-free.
//!
//! Determinism: FxHash is a pure function of the key bytes (no random seed), so unlike `std`'s
//! randomly-seeded `HashMap` its iteration order is fixed run-to-run. Nothing in the front-end may
//! iterate one of these maps to *produce* observable output order. The swap away from SipHash was
//! output-neutral, because the byte-identical `--emit=mir` determinism gate already passed while
//! these maps were randomly seeded. But that argument only covers the swap: LANDMINE — with a fixed
//! seed the gate (`crates/wukongc/tests/determinism.rs`, three fresh processes) can no longer detect
//! an iteration- or insertion-order dependency introduced in an `FxHash*` map, and the file records
//! that caveat itself. Where order matters, sort explicitly or use an ordered container.
//!
//! This mirrors the optimizer's private `fxhash` (kept separate so the crates stay decoupled); the
//! two must agree on the algorithm only insofar as both want a fast integer hash — neither's bits
//! are observable.

use std::hash::{BuildHasherDefault, Hasher};

/// A `HashMap` using [`FxHasher`]. Drop-in for `std::collections::HashMap`.
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;
/// A `HashSet` using [`FxHasher`].
pub type FxHashSet<K> = std::collections::HashSet<K, BuildHasherDefault<FxHasher>>;

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
const ROTATE: u32 = 5;

/// The FxHash state: a single running `u64`, mixed by rotate-xor-multiply per word.
#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(ROTATE) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[..8]);
            self.add(u64::from_le_bytes(b));
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[..4]);
            self.add(u32::from_le_bytes(b) as u64);
            bytes = &bytes[4..];
        }
        for &byte in bytes {
            self.add(byte as u64);
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // Rotate so the well-mixed high bits also land in the low bits the table indexes on.
        self.hash.rotate_left(26)
    }
}
