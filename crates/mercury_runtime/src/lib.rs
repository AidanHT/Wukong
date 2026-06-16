//! `mercury_runtime` — the minimal runtime that Mercury programs link against.
//!
//! Kept deliberately tiny and allocation-explicit, matching the language's philosophy. The
//! interpreter calls these implementations directly; native (LLVM) builds link the same logic
//! compiled as a static library. Today it provides a bump [`Arena`] and a CPU [`parallel_for`].

/// A bump (arena) allocator over an owned byte buffer. Allocation is a pointer bump; freeing is
/// all-at-once via [`Arena::reset`]. This is the idiomatic allocator for kernel scratch space:
/// no per-object bookkeeping, no fragmentation.
pub struct Arena {
    buf: Vec<u8>,
    offset: usize,
}

impl Arena {
    /// Create an arena with `capacity` bytes of backing store.
    pub fn with_capacity(capacity: usize) -> Arena {
        Arena { buf: vec![0u8; capacity], offset: 0 }
    }

    /// Bytes handed out so far.
    pub fn used(&self) -> usize {
        self.offset
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Allocate `size` bytes aligned to `align` (a power of two). Returns the byte offset of the
    /// allocation, or `None` if the arena is exhausted.
    pub fn alloc(&mut self, size: usize, align: usize) -> Option<usize> {
        debug_assert!(align.is_power_of_two());
        let start = (self.offset + align - 1) & !(align - 1);
        let end = start.checked_add(size)?;
        if end > self.buf.len() {
            return None;
        }
        self.offset = end;
        Some(start)
    }

    /// A mutable view of a previously-allocated region.
    pub fn slice_mut(&mut self, offset: usize, len: usize) -> &mut [u8] {
        &mut self.buf[offset..offset + len]
    }

    /// Free everything at once.
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

/// Run `body(i)` for `i` in `lo, lo+step, ...` while `i < hi`.
///
/// The execution order is deterministic and sequential, so it matches the interpreter and serves
/// as a correct reference; a work-stealing thread pool can replace the body later without changing
/// observable results for associative reductions.
pub fn parallel_for(lo: i64, hi: i64, step: i64, mut body: impl FnMut(i64)) {
    if step <= 0 {
        return;
    }
    let mut i = lo;
    while i < hi {
        body(i);
        i += step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alignment_and_exhaustion() {
        let mut a = Arena::with_capacity(64);
        let p0 = a.alloc(1, 1).unwrap();
        assert_eq!(p0, 0);
        // next 8-aligned allocation rounds up past the 1-byte one.
        let p1 = a.alloc(16, 8).unwrap();
        assert_eq!(p1, 8);
        assert_eq!(a.used(), 24);
        // too big -> None, and the arena is unchanged.
        assert!(a.alloc(1024, 8).is_none());
        assert_eq!(a.used(), 24);
        a.reset();
        assert_eq!(a.used(), 0);
    }

    #[test]
    fn parallel_for_sequential_reduction() {
        let mut sum = 0i64;
        parallel_for(0, 10, 2, |i| sum += i);
        assert_eq!(sum, 0 + 2 + 4 + 6 + 8);
    }
}
