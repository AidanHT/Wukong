//! **Run-time turbofish dimensions** — `f::<n>(..)` reads `n` once, into a slot of its own.
//!
//! Semantic analysis gives the value a local holds when `f::<n>(..)` is called a fresh dimension
//! symbol (`wukong_sema`'s `turbofish.rs`, published as `SemaResult::dim_snapshots`): the call's
//! result type, and every shape derived from it, names that symbol rather than `n`. Here the symbol
//! gets its storage. When the call is lowered — before its arguments, the source order of a turbofish
//! written ahead of them — each such local is read and stored into an `i64` slot bound under the
//! fresh symbol in the function's outermost scope. Every later use of a shape that names it (an
//! index stride or bound, `t.dim(k)`, a hidden dimension argument of a further call) reads that slot
//! through the ordinary [`FnLowerer::dim_var_value`] path, and no assignment or shadowing of `n`
//! reaches it.
//!
//! The dimension used to be the local itself, resolved by name at each use: after `n = 5;` a
//! `Tensor[i32, n]` returned for `n = 3` reported `dim(0) == 5`, and reading it to that extent ran
//! past the row (the interpreter trapped, native read whatever lay there).

use wukong_ast::NodeId;
use wukong_mir::{MirType, Op, ValueId};
use wukong_span::Symbol;

use crate::FnLowerer;

impl FnLowerer<'_> {
    /// Read the run-time turbofish values of the call whose callee is `callee_id`, each into the
    /// slot its dimension names from here on. Called before the call's arguments are lowered. A
    /// local that is not bound here leaves its slot unwritten, and the call's hidden dimension
    /// argument then reports itself unresolvable (`call_dim_args`).
    pub(crate) fn snapshot_turbofish_dims(&mut self, callee_id: NodeId) {
        let sema = self.sema;
        let Some(snaps) = sema.dim_snapshots.get(&callee_id) else {
            return;
        };
        let mut read: Vec<Symbol> = Vec::with_capacity(snaps.len());
        for snap in snaps {
            // `f::<n, n>(..)` reads `n` once.
            if read.contains(&snap.dim) {
                continue;
            }
            read.push(snap.dim);
            let Some((src, ty)) = self.lookup(snap.source) else {
                continue;
            };
            let v = self.builder.build(ty.clone(), Op::Load(src, ty.clone()));
            let v = self.coerce_to(v, &ty, &MirType::I64, !snap.unsigned);
            let slot = self.snapshot_slot(snap.dim);
            self.builder.build_void(Op::Store {
                ptr: slot,
                value: v,
            });
        }
    }

    /// The slot the run-time dimension `dim` is read into, made on first use: an `i64` in the
    /// entry block, zeroed there, and bound under `dim` in the function's outermost scope, which no
    /// source name can shadow (`dim` is a fresh symbol). The zero is for the one way a shape can
    /// reach a use without passing its call — an `if`/`match` whose first arm is the call and whose
    /// other arm is a `?` tensor gives the whole merge the first arm's shape — so that use reads a
    /// defined value on every backend rather than an uninitialized slot. It is NOT the `?` tensor's
    /// extent: the slot is zeroed once and never reset, so the use reads 0 only while no call has
    /// written it, and otherwise the last value any call wrote (an earlier loop iteration's, or
    /// another read's of the same immutable binding, which shares the slot). A loop bounded by it
    /// can run past the `?` tensor (`docs/roadmap.md`, the run-time turbofish sharp edge).
    fn snapshot_slot(&mut self, dim: Symbol) -> ValueId {
        if let Some((slot, _)) = self.scopes.first().and_then(|s| s.get(&dim)) {
            return *slot;
        }
        let slot = self.builder.alloca(MirType::I64);
        let here = self.builder.current();
        let entry = self.builder.entry();
        self.builder.switch_to(entry);
        let zero = self
            .builder
            .build(MirType::I64, Op::ConstInt(0, MirType::I64));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: zero,
        });
        self.builder.switch_to(here);
        self.scopes[0].insert(dim, (slot, MirType::I64));
        slot
    }

    /// The run-time dimension the call whose callee is `callee_id` gives its dimension generic
    /// `generic`, if its turbofish named a local there.
    pub(crate) fn turbofish_snapshot(&self, callee_id: NodeId, generic: Symbol) -> Option<Symbol> {
        self.sema
            .dim_snapshots
            .get(&callee_id)?
            .iter()
            .find(|s| s.generic == generic)
            .map(|s| s.dim)
    }
}

#[cfg(test)]
mod tests {
    use wukong_diag::Diagnostic;
    use wukong_mir::{Function, Op, Program, ValueId};
    use wukong_span::{FxHashMap as HashMap, Interner, SourceId};

    fn lower(src: &str) -> (Program, Vec<Diagnostic>, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (prog, diags) = crate::lower_program(&module, &sema, &mut interner);
        (prog, diags, interner)
    }

    const ROW: &str = "fn row<M, N>(a: Tensor[i32, M, N], i: i64) -> Tensor[i32, N] { \
                       return a[i]; } ";

    /// The defining op of every value of `f`.
    fn defs(f: &Function) -> HashMap<ValueId, &Op> {
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter_map(|i| i.result.map(|r| (r, &i.op)))
            .collect()
    }

    /// The callee gets the value read into the snapshot slot, not a fresh load of `n`, and the
    /// slot is written from a load of `n`'s own slot before the call: the stride and extent of the
    /// result read that slot, which the later `n = 1` does not touch.
    #[test]
    fn the_callee_gets_the_value_read_into_its_own_slot() {
        let src = format!(
            "{ROW}fn main() -> i32 {{ let buf: [i32; 6] = [1, 2, 3, 4, 5, 6]; let mut n: i64 = 0; \
             n = n + 3; let r = row::<2, n>(buf, 0); n = 1; print(n); return r[0]; }}"
        );
        let (prog, diags, interner) = lower(&src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        let main = prog
            .funcs
            .iter()
            .find(|f| interner.resolve(f.name) == "main")
            .expect("main is lowered");
        let defs = defs(main);
        let insts: Vec<&Op> = main
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .map(|i| &i.op)
            .collect();
        let row_sym = interner.get("row").expect("`row` is interned");
        let args = insts
            .iter()
            .find_map(|op| match op {
                Op::Call { func, args } if *func == row_sym => Some(args.clone()),
                _ => None,
            })
            .expect("the call to `row`");
        // Hidden dimensions lead: `M`, then `N`, which is a load of the snapshot slot.
        let Some(Op::Load(snap, _)) = defs.get(&args[1]).copied() else {
            panic!("`N` is not a load: {:?}", defs.get(&args[1]));
        };
        assert!(matches!(defs.get(snap), Some(Op::Alloca { .. })));
        // The slot is written twice: the entry block's zero, and a load of another slot (`n`'s).
        let writes: Vec<ValueId> = insts
            .iter()
            .filter_map(|op| match op {
                Op::Store { ptr, value } if ptr == snap => Some(*value),
                _ => None,
            })
            .collect();
        assert_eq!(writes.len(), 2, "{insts:?}");
        assert!(matches!(defs.get(&writes[0]), Some(Op::ConstInt(0, _))));
        let Some(Op::Load(n_slot, _)) = defs.get(&writes[1]).copied() else {
            panic!("the snapshot is not a read: {:?}", defs.get(&writes[1]));
        };
        assert_ne!(n_slot, snap);
        // `n` is written again after the read; the snapshot slot is not.
        assert!(
            insts
                .iter()
                .filter(|op| matches!(op, Op::Store { ptr, .. } if ptr == n_slot))
                .count()
                >= 3,
            "{insts:?}"
        );
    }
}
