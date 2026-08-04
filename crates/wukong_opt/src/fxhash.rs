//! A fast, dependency-free hasher for the optimizer's internal maps.
//!
//! `std`'s default `HashMap`/`HashSet` use SipHash: DoS-resistant, but slow — a fixed per-key cost
//! that dominates when the keys are tiny and the maps are hit millions of times. Every pass here
//! keys its maps on small values drawn from the program being compiled — `ValueId`/`BlockId` inner
//! `u32`s, `(u32,u32)` pairs, `Symbol`s, a `MirType`, or CSE's `Key` (a handful of those in one enum)
//! — never on untrusted network input, so SipHash's collision resistance buys nothing. This is the
//! "FxHash" — rotate, xor, multiply — that `rustc` and Firefox use internally for exactly this
//! reason. It is a few dozen lines, no dependency, and keeps `cargo test` toolchain-free.
//!
//! Determinism: FxHash is a pure function of the key bytes (no random seed), so unlike `std`'s
//! randomly-seeded `HashMap` its iteration order is fixed run-to-run. The optimizer must never
//! iterate one of these maps to *produce* IR order — that was true only of LICM, which now sorts its
//! loop bodies explicitly. So both the hasher swap and its now-fixed iteration order are
//! output-neutral, guarded by the byte-identical `--emit=mir -O2` gate.

use std::hash::{BuildHasherDefault, Hasher};

/// A `HashMap` using [`FxHasher`]. Drop-in for `std::collections::HashMap` on small-integer keys.
pub(crate) type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;
/// A `HashSet` using [`FxHasher`].
pub(crate) type FxHashSet<K> = std::collections::HashSet<K, BuildHasherDefault<FxHasher>>;

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
const ROTATE: u32 = 5;

/// The FxHash state: a single running `u64`, mixed by rotate-xor-multiply per word.
#[derive(Default)]
pub(crate) struct FxHasher {
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
