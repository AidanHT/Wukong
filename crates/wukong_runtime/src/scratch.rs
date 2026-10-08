//! The scratch allocator: the runtime half of the `scratch_<T>(n)` / `scratch_mark()` /
//! `scratch_release(m)` builtins.
//!
//! `alloc_<T>(n)` is a general-purpose heap allocation: one zeroing system allocation per call (for
//! anything large, a fresh mapping whose pages fault in on first touch) and one release per `free`.
//! A kernel that needs a temporary per call — a row buffer, the activation between two layers of a
//! forward pass — pays that round trip and those page faults on every call. The scratch allocator
//! is the temporary-storage design of Odin's `context.temp_allocator`, Jai's temporary storage and
//! Zig's `ArenaAllocator`: a per-thread bump arena whose memory is reused, LIFO, between a mark and
//! its release, so a steady-state `mark; scratch_…; release` loop takes nothing from the system
//! after its first iteration and keeps its working set hot in cache.
//!
//! **The language contract** (the interpreter mirrors it exactly):
//!
//! * `scratch_<T>(n)` returns `n` **zeroed** elements (a negative `n` clamps to 0) at a **64-byte
//!   aligned** address — the same zero-initialization contract as `alloc_<T>`, so the two backends
//!   observe identical contents.
//! * `scratch_mark()` is the **number of live scratch allocations** on this thread: 0 when a run
//!   starts, plus one per `scratch_<T>` call (a zero-length one included). Defining the mark as a
//!   count rather than an address makes it the same value on every backend, so a program that
//!   prints or compares it stays inside the differential contract.
//! * `scratch_release(m)` releases every scratch allocation made after mark `m` was taken; a
//!   negative `m` releases them all and an `m` at or past the current count does nothing. A
//!   released slice is dead — its memory is handed out again — so using one is undefined behaviour
//!   on the native backend, outside the differential contract exactly like a use after `free`.
//!
//! **Memory.** A chain of 64-byte-aligned chunks, each at least twice the size of the largest so
//! far (the first is 1 MiB, or the request if that is larger). Chunks never move — live slices
//! point into them — and are returned to the system only when the thread exits; every chunk past
//! the bump position is free and is reused before anything new is allocated. Nothing is shared
//! between threads, so there is no locking.

use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;

/// Alignment of every scratch allocation (a cache line; what 256- and 512-bit stores want).
const SCRATCH_ALIGN: usize = 64;
/// Size of the first chunk.
const FIRST_CHUNK: usize = 1 << 20;

/// One system allocation backing scratch memory.
struct Chunk {
    base: *mut u8,
    cap: usize,
}

impl Chunk {
    /// A fresh, uninitialized chunk of `cap` bytes (`cap > 0`, a multiple of [`SCRATCH_ALIGN`]), or
    /// `None` when the layout is invalid or the system allocator is exhausted.
    fn new(cap: usize) -> Option<Chunk> {
        let layout = Layout::from_size_align(cap, SCRATCH_ALIGN).ok()?;
        // SAFETY: `cap` is non-zero (callers pass at least `FIRST_CHUNK`) and the alignment is a
        // power of two, so `layout` is a valid non-zero-size layout.
        let base = unsafe { alloc(layout) };
        // Not `then_some`: that would build (and then drop, deallocating null) a `Chunk` on failure.
        if base.is_null() {
            None
        } else {
            Some(Chunk { base, cap })
        }
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        // SAFETY: `base` came from `alloc` with exactly this size and alignment (`Chunk::new`).
        unsafe {
            dealloc(
                self.base,
                Layout::from_size_align_unchecked(self.cap, SCRATCH_ALIGN),
            )
        }
    }
}

/// One thread's scratch arena.
#[derive(Default)]
struct ScratchArena {
    chunks: Vec<Chunk>,
    /// The bump position: a chunk index and a byte offset into it. Every chunk after `cur` is free.
    cur: usize,
    off: usize,
    /// One entry per live allocation, oldest first: the bump position just before it was made.
    marks: Vec<(usize, usize)>,
}

impl ScratchArena {
    /// Allocate `bytes` zeroed bytes, 64-byte aligned. `None` is a size no allocation can satisfy;
    /// it — like a zero-byte request or an exhausted system allocator — yields null, and every call
    /// counts as one live allocation either way, so the mark arithmetic is backend-independent.
    fn alloc(&mut self, bytes: Option<usize>) -> *mut u8 {
        self.marks.push((self.cur, self.off));
        let bytes = match bytes {
            Some(b) if b > 0 => b,
            _ => return std::ptr::null_mut(),
        };
        loop {
            if let Some(chunk) = self.chunks.get(self.cur) {
                let start = self.off.next_multiple_of(SCRATCH_ALIGN);
                if start <= chunk.cap && bytes <= chunk.cap - start {
                    self.off = start + bytes;
                    // SAFETY: `[start, start + bytes)` lies inside this live chunk.
                    unsafe {
                        let p = chunk.base.add(start);
                        std::ptr::write_bytes(p, 0, bytes);
                        return p;
                    }
                }
                // The current chunk is full: step into the next (free) one if it is big enough.
                if self
                    .chunks
                    .get(self.cur + 1)
                    .is_some_and(|next| next.cap >= bytes)
                {
                    self.cur += 1;
                    self.off = 0;
                    continue;
                }
            }
            // Grow. Every chunk after the current one is free and too small for this request, so
            // replace them with one at least twice the largest so far and big enough for it.
            let grown = self
                .chunks
                .iter()
                .map(|c| c.cap)
                .max()
                .unwrap_or(0)
                .saturating_mul(2);
            let Some(cap) = bytes
                .max(FIRST_CHUNK)
                .max(grown)
                .checked_next_multiple_of(SCRATCH_ALIGN)
            else {
                return std::ptr::null_mut();
            };
            let Some(chunk) = Chunk::new(cap) else {
                return std::ptr::null_mut();
            };
            let keep = if self.chunks.is_empty() {
                0
            } else {
                self.cur + 1
            };
            self.chunks.truncate(keep);
            self.chunks.push(chunk);
            self.cur = self.chunks.len() - 1;
            self.off = 0;
        }
    }

    /// The live-allocation count, i.e. the current mark.
    fn mark(&self) -> i64 {
        self.marks.len() as i64
    }

    /// Release every allocation made after `mark` was taken.
    fn release(&mut self, mark: i64) {
        let m = usize::try_from(mark.max(0)).unwrap_or(usize::MAX);
        if m < self.marks.len() {
            let (cur, off) = self.marks[m];
            self.marks.truncate(m);
            self.cur = cur;
            self.off = off;
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<ScratchArena> = RefCell::new(ScratchArena::default());
}

/// `scratch_<T>(n)`: `count` zeroed elements of `elem_size` bytes from this thread's scratch arena,
/// 64-byte aligned. `elem_is_float` is part of the shared call ABI (the interpreter's typed-zero
/// slot model reads it) and is ignored here. Null for a zero-length request, a byte count past
/// `isize::MAX`, or an exhausted system allocator; the call counts toward `scratch_mark()` either
/// way.
#[no_mangle]
pub extern "C" fn wukong_rt_scratch_alloc(
    count: i64,
    elem_size: i64,
    _elem_is_float: i64,
) -> *mut u8 {
    let bytes = (count.max(0) as u128).saturating_mul(elem_size.max(0) as u128);
    let bytes = (bytes <= isize::MAX as u128).then_some(bytes as usize);
    SCRATCH.with(|s| s.borrow_mut().alloc(bytes))
}

/// `scratch_mark()`: the number of live scratch allocations on this thread.
#[no_mangle]
pub extern "C" fn wukong_rt_scratch_mark() -> i64 {
    SCRATCH.with(|s| s.borrow().mark())
}

/// `scratch_release(mark)`: release every scratch allocation made after `mark` was taken.
#[no_mangle]
pub extern "C" fn wukong_rt_scratch_release(mark: i64) {
    SCRATCH.with(|s| s.borrow_mut().release(mark));
}

/// Release every scratch allocation on this thread (keeping its chunks for reuse), so the next run
/// on it starts at mark 0 exactly as a fresh interpreter does. For hosts that run several programs
/// on one thread (the JIT's timing entry); a program never needs it.
pub fn scratch_reset() {
    SCRATCH.with(|s| s.borrow_mut().release(0));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_at(p: *mut u8, n: usize) -> Vec<u8> {
        // SAFETY: the tests only read live scratch allocations of at least `n` bytes.
        unsafe { std::slice::from_raw_parts(p, n).to_vec() }
    }

    #[test]
    fn allocations_are_zeroed_aligned_and_counted() {
        scratch_reset();
        assert_eq!(wukong_rt_scratch_mark(), 0);
        let a = wukong_rt_scratch_alloc(3, 4, 1);
        let b = wukong_rt_scratch_alloc(5, 8, 0);
        assert_eq!(wukong_rt_scratch_mark(), 2);
        for p in [a, b] {
            assert!(!p.is_null());
            assert_eq!(p as usize % 64, 0, "scratch memory must be 64-byte aligned");
        }
        assert_eq!(bytes_at(a, 12), vec![0u8; 12]);
        assert_eq!(bytes_at(b, 40), vec![0u8; 40]);
        // Disjoint: `b` starts at the next 64-byte boundary after `a`'s 12 bytes.
        assert_eq!(b as usize - a as usize, 64);
        scratch_reset();
        assert_eq!(wukong_rt_scratch_mark(), 0);
    }

    #[test]
    fn release_reuses_memory_lifo_and_rezeroes_it() {
        scratch_reset();
        let keep = wukong_rt_scratch_alloc(4, 4, 1);
        let m = wukong_rt_scratch_mark();
        assert_eq!(m, 1);
        let t = wukong_rt_scratch_alloc(16, 4, 1);
        // SAFETY: `t` holds 64 live bytes.
        unsafe { std::ptr::write_bytes(t, 0xAB, 64) };
        wukong_rt_scratch_release(m);
        assert_eq!(wukong_rt_scratch_mark(), 1);
        let again = wukong_rt_scratch_alloc(16, 4, 1);
        assert_eq!(again, t, "the released bytes are handed out again, LIFO");
        assert_eq!(bytes_at(again, 64), vec![0u8; 64], "and zeroed again");
        assert!(!keep.is_null());
        scratch_reset();
    }

    #[test]
    fn release_clamps_its_mark_and_ignores_a_future_one() {
        scratch_reset();
        for _ in 0..3 {
            wukong_rt_scratch_alloc(1, 1, 0);
        }
        wukong_rt_scratch_release(7);
        assert_eq!(
            wukong_rt_scratch_mark(),
            3,
            "a mark past the count releases nothing"
        );
        wukong_rt_scratch_release(3);
        assert_eq!(wukong_rt_scratch_mark(), 3);
        wukong_rt_scratch_release(-5);
        assert_eq!(
            wukong_rt_scratch_mark(),
            0,
            "a negative mark releases everything"
        );
    }

    #[test]
    fn degenerate_requests_are_null_but_still_counted() {
        scratch_reset();
        assert!(wukong_rt_scratch_alloc(0, 4, 1).is_null());
        assert!(wukong_rt_scratch_alloc(-3, 4, 1).is_null());
        assert!(wukong_rt_scratch_alloc(i64::MAX, i64::MAX, 0).is_null());
        assert_eq!(wukong_rt_scratch_mark(), 3);
        scratch_reset();
    }

    #[test]
    fn growing_into_a_new_chunk_never_moves_a_live_allocation() {
        scratch_reset();
        let small = wukong_rt_scratch_alloc(8, 1, 0);
        // SAFETY: `small` holds 8 live bytes.
        unsafe { std::ptr::write_bytes(small, 7, 8) };
        // Larger than the first chunk: forces a second, bigger one.
        let big = wukong_rt_scratch_alloc(3 << 20, 1, 0);
        assert!(!big.is_null());
        assert_eq!(big as usize % 64, 0);
        assert_eq!(
            bytes_at(small, 8),
            vec![7u8; 8],
            "the first chunk stays put"
        );
        assert_eq!(bytes_at(big, 4096), vec![0u8; 4096]);
        // Released and re-requested, the big block comes back from the cached chunk.
        wukong_rt_scratch_release(1);
        let again = wukong_rt_scratch_alloc(3 << 20, 1, 0);
        assert_eq!(again, big);
        scratch_reset();
    }
}
