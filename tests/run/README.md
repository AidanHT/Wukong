# `tests/run/` — passing-program fixtures

User-written `.wk` programs that must compile and execute. This directory is the primary
correctness corpus for the language — not `bench/` (measurement harnesses) or `tools/` (scripts and
cloud launchers).

The fixture count is deliberately not written here. It lived in four documents at once and all four
were wrong (363 / 332 / 371 against a tree of 388), because a number in prose has nothing checking
it. The one place it is pinned is `MIN_FIXTURES` in `crates/wukongc/tests/run.rs`, which is a
floor asserted by `collect_programs`: adding fixtures never trips it, losing one does. Raise it in
the same commit that adds fixtures.

Every harness collects fixtures with a **non-recursive** `read_dir` filtered on `extension == "wk"`.
Anything dropped here becomes a gated fixture immediately. Imported support modules live in
[`lib/`](lib/) — that subdirectory is never scanned.

## Directives

Put at the top of each file. Leading whitespace and the spacing after `//` are insignificant; the
directive **name** is not, and a name this harness does not recognize is a hard test failure rather
than silence. The grammar has exactly one reader, `crates/wukongc/tests/common/mod.rs`, shared by
the executing gate and the lint.

```wk
// RUN: --run
// EXPECT-EXIT: 0
// EXPECT-OUT: 8
// EXPECT-OUT-APPROX: 1.6487212707001282 1.625e-7
```

| directive | meaning |
|---|---|
| `// RUN: <args>` | **replaces** the whole arg list (default `--run`). Only `run_suite` honours it; the differential gates build their own args |
| `// EXPECT-EXIT: <n>` | **required on every fixture.** `run_suite` panics without it |
| `// EXPECT-OUT: <line>` | one expected stdout line, exact string, in order |
| `// EXPECT-OUT-VALUES: <token> ...` | one or more whitespace-separated exact tokens, each expecting a successive stdout line |
| `// EXPECT-OUT-APPROX: <value> <rel-tol>` | one expected stdout line, compared numerically inside a declared two-sided relative tolerance |
| `// EXPECT-OUT-NONE` | stdout must be empty — a positive assertion, for a fixture that prints nothing |
| `// EXPECT-STDERR: <substring>` | stderr must contain this substring (`run_suite` only) |

**Every fixture must declare `EXPECT-EXIT` and one of `EXPECT-OUT` / `EXPECT-OUT-VALUES` / `EXPECT-OUT-APPROX` /
`EXPECT-OUT-NONE`.** Both of `run_suite`'s checks used to be conditional on their directive being
present, so a fixture declaring neither ran and asserted nothing — which is how
`gemm_fused_bias_slice.wk` printed 36 unchecked numbers for its whole life, and how `assert_fail.wk`
stayed green for a program that had stopped compiling (`COMPILE_ERROR` is also exit 1 with empty
stdout).

`// EXPECT-OUT-VALUES: 1 -2 0.25` is shorthand for three `EXPECT-OUT` directives,
expecting `1`, `-2`, and `0.25` on three successive stdout lines. Multiple directives
append expectations in source order and can mix with `EXPECT-OUT` and `EXPECT-OUT-APPROX`.
The tokens are exact strings: `1.0` does not match `1`, and actual stdout is not split
into tokens. Use `EXPECT-OUT` for an empty line or a line containing whitespace.
There is no quoting or escaping syntax in `EXPECT-OUT-VALUES`; an empty token list is
an error, as is combining any output expectation with `EXPECT-OUT-NONE`.

## Floats: exactly two legal forms

A float assertion is either **exact** or **declared**. Nothing else.

- **Exact** — `EXPECT-OUT` (or its `EXPECT-OUT-VALUES` shorthand) against a raw `print(x)` whose value is exactly representable in f32
  (integers, halves, quarters, 0.25, -0.5, 100.0) and order-independent. Preferred: it has no
  tolerance at all.
- **Declared** — `EXPECT-OUT-APPROX` with a written relative tolerance, for a value the scalar
  semantics do not pin to a bit pattern (a transcendental, a reassociating reduction over inexact
  data). The tolerance must come from a real source — a kernel's published error bound, an
  exhaustive measurement — and that source must be written into `DECLARED_TOLERANCES` in
  `crates/wukongc/tests/corpus_lint.rs`. A tolerance token with no entry there fails the lint.
  `corpus_lint.rs::no_approx_line_in_the_corpus_is_vacuous` additionally proves each line rejects a
  value ten tolerances away, so an APPROX line that asserts nothing cannot ship.

### The third form, which is legacy and should not spread

Most of this corpus prints `(x * 1000.0) as i32`. That cast is a **truncation toward zero**, so an
expectation `K` accepts the half-open interval `[K/S, (K+1)/S)`: a one-sided, undeclared tolerance
of one whole step, and 0 tolerance on the other side. On raw (unscaled) prints it is as loose as
100% relative. A handful of files instead write `((x * S) + 0.5) as i32`, which is symmetric
rounding — `softmax.wk` uses both conventions on adjacent lines, over lanes of the same buffer.
Neither is stated anywhere in the fixture and neither is machine-readable, which is why an agent
told "do not widen an existing tolerance" cannot tell what tolerance it is looking at.

New fixtures use one of the two legal forms. Converting an existing one is a tightening and must
say so in the commit body, with the old interval and the new one both written out.

## Authoring rules

1. **Print every lane, never a reduction.** A fixture that prints one sum lets a partially-correct
   buffer pass, and a sum is invariant under every permutation of the output. Print each element,
   and choose `n` so a vector/tail boundary lands inside the expectations
   (`vhorner_vector_and_tail.wk` uses n=13 = one 8-lane group + a 5-element tail;
   `vmath_exp_every_lane.wk` uses n=19 = two groups + a 3-element tail).
2. **Expected values come from scalar semantics, computed independently — never a recording of
   compiler output.** Say so in the prose, and say *how*: the closed form, the f64 reference, the
   step-by-step f32 evaluation. Pasting observed stdout pins the bug — `transcendental_kernel.wk`'s
   expectation is one ULP above the true `exp(0.5)` for exactly that reason, and its prose now
   records that it is a snapshot rather than a derivation.
3. **Recognizer/kernel fixtures are GATE-BLIND.** The kernel fires identically on both backends at
   every opt level, so `native_matches_interpreter` and the invariance gates cannot see whether it
   fired at all. Verify by hand with
   `cargo run -q -p wukongc -- --emit=mir -O2 tests/run/<f>.wk | grep wukong_`, name the symbol you
   expect in the prose, and check `tests/recognizer-census.txt` — every fixture's dispatched
   symbols are pinned there, so a recognizer that stops firing fails
   `cargo test -p wukongc --test recognizer_census`.
4. **Poison an output buffer whose expectation could be zero.** `alloc_f32` is `alloc_zeroed` and
   array literals are written `[0.0; N]`, so a `0` lane cannot distinguish a correct kernel from one
   that declined and wrote nothing — and `0` is the single most common expected line in the corpus.
   Fill the buffer with a sentinel (`-12345.0`) before the call, or choose data where no correct
   output lane is 0. `linear_bias_relu.wk` and `gemm_fused_bias_slice.wk` show the first.
5. **Do not let the input's period divide the kernel's block size.** A reduction over data that
   cycles with period 4 gives every 16-element group the same sum, so a kernel that skipped one
   group and processed another twice — wrong base, wrong stride, duplicated or dropped block —
   returns the right total. Pick a period coprime to the block (7 against a 16-wide step), and give
   a dot product's two operands *different* periods, or the second operand's indexing is asserted by
   nothing. `reduce_bf16_shapes.wk` documents the arithmetic.
6. **Use values whose decimal rendering is unambiguous** on both backends. Never a wall-clock or
   otherwise nondeterministic value — it breaks all four differential gates.
7. **File-I/O writers need unique `io_test_*.bin` names** (the harness uses per-run scratch CWDs,
   but the fixtures share a naming space).

Rules 1–3 are judgement calls no lint can make. Rules 4 and 5 are too. What *is* mechanically
enforced lives in `crates/wukongc/tests/corpus_lint.rs`, and it runs inside
`cargo test --workspace`.

## `lang_*` — the thesis fixtures

`lang_generic_saxpy.wk` and `lang_generic_naive_gemm.wk` each carry the **same body twice**, once under
`@generic` and once without, and print every lane of both. The pair exists because this corpus cannot
otherwise see a recognizer: both spellings compute the same bytes, so the proof that `@generic`
suppressed the kernel — and that it suppressed it for **one function**, not the module — is a MIR count
asserted in `crates/wukongc/tests/recognize.rs`:

```sh
cargo run -q -p wukongc -- --emit=mir -O0 tests/run/lang_generic_naive_gemm.wk   # exactly one sgemm
```

A new `lang_*` fixture should keep that shape: a written-out loop, an un-attributed twin where one is
meaningful, every lane printed, and a MIR witness in `recognize.rs` rather than a behavioral claim.

## Run one fixture

```sh
cargo run -q -p wukongc -- --run -O2 tests/run/foo.wk
cargo run -q -p wukongc -- --emit=mir -O2 tests/run/foo.wk | grep wukong_   # what it dispatches
```
