//! `[agg; n]` — the repeat initializer of an array whose element is itself an aggregate (a struct,
//! a tuple, a data-carrying enum or a nested array, all byte-buffer `MirType::Array` elements).

use wukong_ast::Expr;
use wukong_mir::{MirType, ValueId};

use crate::{mir_byte_size, FnLowerer};

impl FnLowerer<'_> {
    /// Initialize the `n` aggregate elements (`elem`) of the array at `base` from `[value; n]`.
    ///
    /// `value` is evaluated exactly **once**, into element 0, and elements `1..n` are deep copies
    /// of it — the rule the scalar arm of [`FnLowerer::lower_array_init`] already followed. The
    /// previous lowering re-lowered the initializer expression once per element, which (a) ran a
    /// side-effecting initializer `n` times — `[make(); 3]` called `make` three times, on every
    /// backend, so no differential gate could see it — and (b) emitted `n` copies of its IR with no
    /// bound, so IR size grew with the array's length. The copies follow the same unroll policy as
    /// every other aggregate copy ([`FnLowerer::unroll_aggregate`]): straight-line when small (or
    /// when a gradient candidate needs it, up to its byte bound), a compact counted loop otherwise.
    pub(super) fn lower_aggregate_repeat(
        &mut self,
        base: ValueId,
        elem: &MirType,
        n: u32,
        value: &Expr,
    ) {
        let ety = self.expr_ty(value);
        if n == 0 {
            // No element to hold it, but the initializer is still evaluated once, exactly as the
            // scalar arm evaluates `value` before deciding how many slots to fill.
            let scratch = self.builder.alloca(elem.clone());
            self.init_field(scratch, &ety, value);
            return;
        }
        let first = self.gep_elem(base, elem, 0);
        self.init_field(first, &ety, value);
        let rest = u64::from(n - 1);
        if rest == 0 {
            return;
        }
        let bytes = mir_byte_size(elem).saturating_mul(rest);
        if self.unroll_aggregate(rest, Some(bytes)) {
            for i in 1..=rest as i128 {
                let dst = self.gep_elem(base, elem, i);
                self.emit_copy(dst, first, &ety);
            }
        } else {
            // Element `1 + i` for `i` in `0..rest`: index from element 1's address so the loop
            // counter needs no offset arithmetic (an element-typed GEP is additive on both
            // backends — native scales by the byte size, the interpreter by the slot count).
            let second = self.gep_elem(base, elem, 1);
            self.lower_counted_loop(rest, |this, i| {
                let dst = this.gep_elem_value(second, elem, i);
                this.emit_copy(dst, first, &ety);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{lower_program_with, LowerOptions};
    use wukong_mir::{Function, Op, Program};
    use wukong_span::{Interner, SourceId};

    fn lower(src: &str, opts: LowerOptions) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (program, diags) = lower_program_with(&module, &sema, &mut interner, opts);
        assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
        for f in &program.funcs {
            let errors = wukong_mir::verify::verify_function(f);
            assert!(errors.is_empty(), "{errors:?}");
        }
        (program, interner)
    }

    fn find_fn<'p>(program: &'p Program, interner: &Interner, name: &str) -> &'p Function {
        program
            .funcs
            .iter()
            .find(|f| interner.resolve(f.name) == name)
            .unwrap_or_else(|| panic!("no function `{name}`"))
    }

    fn calls_to(f: &Function, interner: &Interner, callee: &str) -> usize {
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(&i.op, Op::Call { func, .. } if interner.resolve(*func) == callee))
            .count()
    }

    fn inst_count(f: &Function) -> usize {
        f.blocks.iter().map(|b| b.insts.len()).sum()
    }

    const POINT: &str = "struct P { x: i32, y: i32 }
        fn mk(v: i32) -> P { print(v); return P { x: v, y: v * 10 }; }";

    /// `[agg; n]` evaluates its initializer once, at every length and on both copy shapes (the
    /// straight-line copies of a short repeat and the counted loop of a long one). It used to call
    /// `mk` once per element.
    #[test]
    fn aggregate_repeat_evaluates_its_initializer_once() {
        for n in [1u32, 3, 8, 9, 64] {
            let src = format!(
                "{POINT}
                 fn rep() -> i32 {{ let a: [P; {n}] = [mk(5); {n}]; return a[{last}].y; }}",
                last = n - 1
            );
            for unroll_for_autodiff in [true, false] {
                let (program, interner) = lower(
                    &src,
                    LowerOptions {
                        unroll_for_autodiff,
                        ..LowerOptions::default()
                    },
                );
                let rep = find_fn(&program, &interner, "rep");
                assert_eq!(calls_to(rep, &interner, "mk"), 1, "n = {n}");
            }
        }
    }

    /// A gradient candidate (a float-returning function) still unrolls its aggregate copies so it
    /// stays single-block, but only up to `AUTODIFF_UNROLL_MAX_BYTES`: a 4096-element copy used to
    /// emit 4096 straight-line load/store pairs (~24k instructions) and is now a compact loop.
    #[test]
    fn gradient_candidate_copy_unroll_is_bounded() {
        let src = "fn big() -> f32 {
                let a: [f32; 4096] = [1.5; 4096];
                let b = a;
                return b[4095];
            }";
        let (program, interner) = lower(src, LowerOptions::default());
        let big = find_fn(&program, &interner, "big");
        assert!(
            inst_count(big) < 512,
            "a 4096-element copy must not unroll: {} instructions",
            inst_count(big)
        );
        let repeat = "struct Q { a: f32, b: f32 }
            fn fill() -> f32 { let q: [Q; 4096] = [Q { a: 1.0, b: 2.0 }; 4096]; return q[7].b; }";
        let (program, interner) = lower(repeat, LowerOptions::default());
        let fill = find_fn(&program, &interner, "fill");
        assert!(
            inst_count(fill) < 512,
            "a 4096-element `[agg; n]` must not unroll: {} instructions",
            inst_count(fill)
        );
    }

    /// Below the bound, the single-block form is kept for gradient candidates — and only for them:
    /// with `unroll_for_autodiff` cleared (an ordinary compile) the same copy is the counted loop.
    #[test]
    fn small_copies_unroll_only_for_autodiff() {
        let src = "fn small() -> f32 {
                let a: [f32; 16] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
                let b = a;
                return b[15];
            }";
        let blocks = |unroll_for_autodiff: bool| {
            let (program, interner) = lower(
                src,
                LowerOptions {
                    unroll_for_autodiff,
                    ..LowerOptions::default()
                },
            );
            find_fn(&program, &interner, "small").blocks.len()
        };
        let candidate = blocks(true);
        let ordinary = blocks(false);
        assert!(
            ordinary > candidate,
            "an ordinary compile lowers the 16-element copy to a loop ({ordinary} blocks vs {candidate})"
        );
    }
}
