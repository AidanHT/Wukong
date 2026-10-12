//! Stack-slot coloring: allocas whose lifetimes never overlap share one Cranelift stack slot, so a
//! function's frame holds the largest set of simultaneously-live slots instead of every slot it
//! declares. The JIT and the object emitter lay frames out through the same lowering
//! ([`crate::FnTranslator`]), so both get it.
//!
//! The lifetimes come from [`wukong_opt::SlotLifetimes`] (see that module for what "live" means and
//! why sharing storage between two slots that are never live together cannot be observed). The
//! interpreter is the language's oracle and keeps one slot per alloca, so the run corpus's
//! interpreter-versus-native differential checks every coloring this makes.
//!
//! # The coloring
//!
//! Greedy interval coloring in a deterministic order: allocas by slot size (largest first, ties in
//! layout order), each joining the first color created so far whose members it does not interfere
//! with, else opening a new one. A color's slot is as large and as aligned as its largest and most
//! aligned member, so taking the big slots first means a later member never grows the slot it
//! joins. Two kinds of alloca never share:
//!
//! * one whose address escapes the analysis (live for the whole function), and one whose byte size
//!   does not fit the backend's layout (its lowering reports `C0002`);
//! * when Cranelift optimizes (source `-O1` and up), two in different **alias regions**. Every
//!   access to a private stack slot carries Cranelift's [`crate::PRIVATE_STACK_REGION`] and every
//!   other access the default region, and Cranelift's alias analysis requires one memory location
//!   never to be accessed under two regions — so a color holds private slots only, or non-private
//!   slots only. Merging the two would mean giving up the private region (and the load forwarding
//!   it buys) for the private members, which costs speed where speed is asked for.
//!
//! At `-O0` Cranelift runs no alias analysis (`opt_level=none` skips the e-graph pass that holds
//! it), so the regions buy nothing there and colors mix freely; a private member of a mixed color is
//! then **demoted** — its accesses carry the default region like every other member's — so the
//! one-region-per-location rule still holds literally ([`SlotPlan::demotes`]).
//!
//! A member computes its address from the shared slot exactly as it would from a slot of its own
//! (an over-aligned one still rounds its base up inside the slot), so a function in which nothing
//! shares lowers to the same instructions and the same frame as without coloring.
//!
//! `WUKONG_STACK_COLOR=0` turns coloring off (every alloca gets its own slot, as before) — an A/B
//! knob and a kill switch; it can only change frame sizes, never a computed value.

use std::collections::HashMap;

use wukong_mir::{Function, MirType, Op, ValueId};
use wukong_opt::slot_lifetime::{ranges_intersect, ranges_union};
use wukong_opt::{AliasInfo, CaptureSummary, SlotLifetimes};

use crate::{align_shift, over_aligned_slot_bytes, size_of, FRAME_ALIGN};

/// The explicit stack slot one alloca needs: its size in bytes and its alignment as a power of two.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SlotSpec {
    pub(crate) bytes: u32,
    pub(crate) align_shift: u8,
}

impl SlotSpec {
    /// The smallest slot that can hold either `self` or `other`.
    fn cover(self, other: SlotSpec) -> SlotSpec {
        SlotSpec {
            bytes: self.bytes.max(other.bytes),
            align_shift: self.align_shift.max(other.align_shift),
        }
    }
}

/// The slot an `Op::Alloca` of `bytes` bytes (at least 1) with `@align` request `align` (`0`:
/// natural) needs, and the alignment `want` its address must have. When `want` exceeds
/// [`FRAME_ALIGN`] the slot carries slack and the lowering rounds the address up inside it (see
/// [`over_aligned_slot_bytes`]); that rounding needs a 16-aligned base, which the slot's own
/// alignment then guarantees.
pub(crate) fn alloca_slot(bytes: u32, align: u32) -> (SlotSpec, u32) {
    let natural = align_shift(bytes);
    let want = (1u32 << natural).max(align);
    let spec = if want > FRAME_ALIGN {
        SlotSpec {
            bytes: over_aligned_slot_bytes(bytes, want),
            align_shift: natural.max(FRAME_ALIGN.trailing_zeros() as u8),
        }
    } else {
        SlotSpec {
            bytes,
            align_shift: want.trailing_zeros() as u8,
        }
    };
    (spec, want)
}

/// Is coloring on? `WUKONG_STACK_COLOR=0` turns it off; a test can override the environment on its
/// own thread (the module is populated on the calling thread, so the override reaches every
/// function, also under parallel codegen).
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = FORCE.with(|f| f.get()) {
        return forced;
    }
    !matches!(
        std::env::var("WUKONG_STACK_COLOR").ok().as_deref(),
        Some("0")
    )
}

#[cfg(test)]
thread_local! {
    static FORCE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with coloring forced on or off for compiles started on this thread.
#[cfg(test)]
pub(crate) fn with_coloring<T>(on: bool, f: impl FnOnce() -> T) -> T {
    let prev = FORCE.with(|c| c.replace(Some(on)));
    let out = f();
    FORCE.with(|c| c.set(prev));
    out
}

/// What coloring needs to know about the program and the code generator.
#[derive(Clone, Copy)]
pub(crate) struct ColorConfig<'a> {
    /// Which calls may retain a pointer, program-wide.
    pub(crate) calls: &'a CaptureSummary,
    /// May private and non-private slots share (the code generator runs no alias analysis)?
    pub(crate) mix_regions: bool,
}

/// Which frame slot each alloca of one function uses.
#[derive(Default)]
pub(crate) struct SlotPlan {
    /// The color of every alloca that shares a slot with another (an alloca absent here gets a
    /// slot of its own).
    color: HashMap<ValueId, u32>,
    /// Each color's slot.
    colors: Vec<SlotSpec>,
    /// Private slots that share with a non-private one: their accesses take the default region.
    demoted: Vec<ValueId>,
}

/// One color while the plan is built.
struct Color {
    spec: SlotSpec,
    /// Every member is a private slot.
    private: bool,
    live: Vec<(u32, u32)>,
    /// Each member, and whether it is private.
    members: Vec<(ValueId, bool)>,
}

impl SlotPlan {
    /// Color the allocas of `f`. `alias` must be `f`'s own analysis (it decides each slot's alias
    /// region); `cfg.calls` summarizes the program `f` belongs to.
    pub(crate) fn new(f: &Function, cfg: ColorConfig<'_>, alias: &AliasInfo) -> SlotPlan {
        let allocas: Vec<(ValueId, &MirType, u32)> = f
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter_map(|inst| match (&inst.op, inst.result) {
                (Op::Alloca { ty, align, .. }, Some(v)) => Some((v, ty, *align)),
                _ => None,
            })
            .collect();
        if allocas.len() < 2 {
            return SlotPlan::default();
        }
        let lifetimes = SlotLifetimes::analyze(f, cfg.calls);
        debug_assert!(lifetimes.allocas().eq(allocas.iter().map(|a| a.0)));
        // Candidates: sized, non-escaping allocas, largest slot first, ties in layout order.
        let mut order: Vec<(usize, SlotSpec)> = allocas
            .iter()
            .enumerate()
            .filter(|&(i, _)| !lifetimes.escapes(i))
            .filter_map(|(i, &(_, ty, align))| {
                size_of(ty).map(|bytes| (i, alloca_slot(bytes.max(1), align).0))
            })
            .collect();
        if order.len() < 2 {
            return SlotPlan::default();
        }
        order.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes).then(a.0.cmp(&b.0)));
        let mut colors: Vec<Color> = Vec::new();
        for (i, spec) in order {
            let v = allocas[i].0;
            let private = alias.is_private_stack(v);
            let live = lifetimes.live_ranges(i);
            match colors.iter_mut().find(|c| {
                (cfg.mix_regions || c.private == private) && !ranges_intersect(&c.live, live)
            }) {
                Some(c) => {
                    c.spec = c.spec.cover(spec);
                    c.live = ranges_union(&c.live, live);
                    c.members.push((v, private));
                    c.private &= private;
                }
                None => colors.push(Color {
                    spec,
                    private,
                    live: live.to_vec(),
                    members: vec![(v, private)],
                }),
            }
        }
        let mut plan = SlotPlan::default();
        for c in colors.into_iter().filter(|c| c.members.len() >= 2) {
            let id = plan.colors.len() as u32;
            plan.colors.push(c.spec);
            for (v, private) in c.members {
                plan.color.insert(v, id);
                if private && !c.private {
                    plan.demoted.push(v);
                }
            }
        }
        plan.demoted.sort_unstable_by_key(|v| v.0);
        plan
    }

    /// Does the private slot `a` share with a non-private one, so that its accesses must carry the
    /// default alias region instead of [`crate::PRIVATE_STACK_REGION`]? Only ever at `-O0`.
    pub(crate) fn demotes(&self, a: ValueId) -> bool {
        self.demoted.binary_search_by_key(&a.0, |v| v.0).is_ok()
    }

    /// The shared slot alloca `v` uses — its color and that color's slot — or `None` when it gets a
    /// slot of its own.
    pub(crate) fn color_of(&self, v: ValueId) -> Option<(u32, SlotSpec)> {
        self.color.get(&v).map(|&c| (c, self.colors[c as usize]))
    }

    /// The number of shared slots.
    pub(crate) fn color_count(&self) -> usize {
        self.colors.len()
    }
}

/// The explicit-slot bytes of one function's frame with and without coloring: each alloca's slot
/// (or each shared slot once), its size rounded up to its alignment. A static measure of what the
/// layout hands Cranelift, before spill slots and the ABI's own frame.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct FrameBytes {
    /// Every alloca in a slot of its own.
    pub separate: u64,
    /// After coloring.
    pub colored: u64,
}

impl std::ops::AddAssign for FrameBytes {
    fn add_assign(&mut self, other: FrameBytes) {
        self.separate += other.separate;
        self.colored += other.colored;
    }
}

fn padded(spec: SlotSpec) -> u64 {
    let align = 1u64 << spec.align_shift;
    u64::from(spec.bytes).div_ceil(align) * align
}

/// [`FrameBytes`] of every function of `program` as the native backend lays it out for source
/// optimization level `opt_level` (see [`crate::BackendOpt`]), summed — the static frame reduction
/// coloring buys (allocas whose size does not fit the layout are left out).
pub fn frame_bytes(program: &wukong_mir::Program, opt_level: u8) -> FrameBytes {
    let calls = CaptureSummary::analyze(program);
    let cfg = ColorConfig {
        calls: &calls,
        mix_regions: opt_level == 0,
    };
    let mut total = FrameBytes::default();
    for f in &program.funcs {
        total += function_frame_bytes(f, cfg);
    }
    total
}

/// [`FrameBytes`] of one function.
pub(crate) fn function_frame_bytes(f: &Function, cfg: ColorConfig<'_>) -> FrameBytes {
    let alias = AliasInfo::analyze(f);
    let plan = SlotPlan::new(f, cfg, &alias);
    let mut out = FrameBytes::default();
    let mut counted = vec![false; plan.color_count()];
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        let (Op::Alloca { ty, align, .. }, Some(v)) = (&inst.op, inst.result) else {
            continue;
        };
        let Some(bytes) = size_of(ty) else { continue };
        let own = padded(alloca_slot(bytes.max(1), *align).0);
        out.separate += own;
        match plan.color_of(v) {
            Some((c, spec)) => {
                if !counted[c as usize] {
                    counted[c as usize] = true;
                    out.colored += padded(spec);
                }
            }
            None => out.colored += own,
        }
    }
    out
}

#[cfg(test)]
mod tests;
