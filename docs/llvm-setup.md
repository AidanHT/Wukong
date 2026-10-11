# The optional LLVM path (`--emit=llvm-ir`)

**You do not need LLVM to build, run, or produce native binaries with Wukong.** The default native
path is **Cranelift** (pure Rust, in-process): `--run --backend=native` JITs, and `--emit=obj` /
`--emit=exe` write a native object / executable with **no LLVM toolchain** (verified on a box with no
`clang` installed — `--emit=obj` emits a COFF/ELF object directly from Cranelift). `--emit=exe` then
links that object into an executable — preferring a **`rustc`-driven link** (rustc drives the
platform's native linker and links `wukong_runtime`), falling back to the system C compiler (`cc`, or
`$CC`) **only when rustc cannot be run at all or no `wukong_runtime` rlib is reachable from the
`wukongc` binary's directory** — that means neither `libwukong_runtime.rlib` beside it (what a plain
`cargo build` uplifts) nor a hash-suffixed `deps/libwukong_runtime-*.rlib` (what a `cargo test` or
`cargo run -p wukongc` tree has instead, since cargo uplifts a library only for a `build`): if rustc
runs and the link *fails*, that is reported as an error (exit 1) and is
deliberately *not* retried with `cc`, so a real link bug is never swallowed. For a program supported
by the C fallback, inability to run either linker produces exit 2 (`UNIMPLEMENTED`) and attempts
to preserve the object. A linker that runs and fails to link produces exit 1.

The `cc` fallback provides integer, bool, float and string printing, assertions, float remainder,
heap operations and scalar trigonometry, and its output is byte-identical to `--run`: floats are
printed exactly as the interpreter's Rust `Display` prints them (the shortest decimal that
round-trips, written positionally, with `inf`/`-inf`/`NaN`/`-0`), derived in C from correctly rounded
`%e` conversions checked with `strtod`, not approximated with `%g`. What the C runtime cannot
provide is the recognized `wukong_*` microkernels — a program that calls one fails to link (loudly,
with a `note:` naming the fallback), and on Windows the MinGW linker also cannot consume an MSVC
object's data relocations (string literals). The preferred rustc path links `wukong_runtime` and has
neither limitation. To run with zero external toolchain, use the interpreter:
`wukongc --run program.wk`.

Scalar sine/cosine use the same bounded polynomial and explicit FMA operations in the Rust and
C runtimes; arguments with magnitude at least 4096 use the platform's full-range `sinf`/`cosf`.
The C link command disables implicit FP contraction so the polynomial's separate operations
retain their rounding. Tangent remains the quotient of the shared sine and cosine.

`--target-cpu=<native|x86-64|x86-64-v2|x86-64-v3|x86-64-v4>` selects the instruction-set level of the
object (default `native`, every feature of the compiling host). A named level is exact and
host-independent, and below `x86-64-v3` the general vectorizer's raw-AVX2 loop kernels are not used,
so `--emit=exe --target-cpu=x86-64` runs on any x86-64 CPU; the linked runtime kernels already pick
their AVX2 path at run time.

LLVM enters **only** through `--emit=llvm-ir`, which prints **textual LLVM IR** for inspection — and
even that needs no LLVM installed (it is just text on stdout):

```sh
wukongc --emit=llvm-ir -O2 examples/fib.wk    # print textual LLVM IR
```

The emitted IR has no fast-math flags. Every contraction the compiler performs is already an
explicit `llvm.fma.*` call, and a `@strict_fp` function (or any function under `--fp=strict`)
contains none. Plain `fmul`/`fadd` are meant to stay separately rounded. If you compile the IR
yourself with `-ffp-contract=fast` or `-ffast-math`, the backend may fuse or reassociate them, and
the results will differ from `--run` (see the language guide, "Floating-point semantics").

The only reason to install LLVM is to take that emitted textual IR and compile it yourself with an
external `clang`/`llc`, or to build a future in-process `inkwell` backend. Neither is required for
`--run`, `--backend=native`, `--emit=obj`, or `--emit=exe`.

### What the emitted module means

The module implements the interpreter's language, not LLVM's defaults:

- float `!=` is the unordered `fcmp une` (NaN `!=` NaN is true); integer division and remainder by
  zero produce 0 and `INT_MIN / -1` wraps; shift counts are masked to the operand width — each is
  an explicit guard, because LLVM makes all three poison;
- `f16`/`bf16` values are computed in `float` registers and rounded onto the half/bfloat grid only
  at constants, casts, loads and stores (the interpreter's round-at-store/cast/load model) — so
  `half`/`bfloat` appear only as memory types. An `f16` rounds with `fptrunc`; a `bf16` rounds with
  the runtime's own integer arithmetic (`wukong_runtime::f32_to_bf16_bits`, NaN payloads included),
  because LLVM's `fptrunc … to bfloat` is a call to `__truncsfbf2`, which neither the Rust runtime
  link nor an MSVC link provides;
- booleans occupy a byte in memory (`i8` storage, low bit significant), never LLVM's bit-packed
  `<n x i1>`;
- a vector load or store promises only its lane's alignment (`align 4` for `<4 x float>`), because
  the vectorizer accesses vectors at every element offset, and a stack slot has the native
  backend's alignment (its size rounded to a power of two, at most 16 bytes, 32 from 32 bytes up);
- float literals are spelled in LLVM's exact hexadecimal form, so infinities, NaNs, tiny values and
  decimals that are inexact in `float` are all valid IR.

It names the host's `target datalayout` and `target triple` (x86-64 Windows: `e-m:w-…` /
`x86_64-pc-windows-msvc`; x86-64 Linux: `e-m:e-…` / `x86_64-unknown-linux-gnu`). The layout is the
string LLVM 15–17 compute for the triple; LLVM 18 and later upgrade it on read to their own (they
add `i128:128`), so the same module is exact for the LLVM 15 behind CI's `llvmlite` verifier and for
the LLVM 19 `clang` installed below. Every external symbol it references is one the `--emit=exe`
link already provides, **declared with the one signature it is defined with** — the link-ABI law,
checked over the whole run corpus by `wukong_codegen_llvm`'s tests against the runtime's Rust
sources:

- `print`/`println`/`assert` lower to the same `wukong_rt_print_{i64,u64,f64,f32,bool,char,str}` /
  `wukong_rt_assert` ABI the Cranelift object calls (with the same argument widening; an `f32`,
  `f16` or `bf16` prints through `wukong_rt_print_f32` at `f32` precision);
- every `wukong_*` runtime entry — recognized kernels, and the `wukong_rt_fmt_*` / `wukong_rt_arg*`
  / `wukong_rt_env` / `wukong_rt_read_stdin` / `wukong_rt_exit` builtins — is declared from its
  `#[no_mangle] extern "C"` definition in `wukong_runtime` (a generated table, never inferred from a
  call site), and each argument is converted to its slot exactly as the Cranelift call converts it:
  integers widened to `i64`, an absent optional pointer passed as `ptr null`;
- the general vectorizer's synthesized kernels — raw AVX2 bytes inside the Cranelift backend — are
  **defined in the module** from their backend-agnostic recipe in their own lane type (`f32`,
  `f64`, `i32` or `i64`), with the interpreter's exact element order and reduction reassociation.

A construct that cannot be expressed faithfully is refused with `C0001` and exit 2 rather than
emitted wrong, and the message names its reason: `wukong_codegen_llvm::DECLINED` lists every one (a
`print` of a vector value, a bitcast to or from `f16`/`bf16`, two incompatible signatures for one
undeclared external symbol, an untyped external function address, a call no runtime definition can
take, and two MIR invariant violations). No run-corpus program or example is refused; `emit.rs`
holds that count at a ratchet of zero.

## Why textual IR (and not inkwell in-process)

The original LLVM plan targeted `inkwell` (the in-process libLLVM API). On Windows that path is
fragile: `llvm-sys` requires the LLVM **development** libraries plus `llvm-config`, which the stock
`LLVM-*-win64.exe` installer does **not** ship, and the Rust (MSVC) / LLVM build ABIs must match.
Emitting textual IR sidesteps all of that — it needs only the LLVM command-line tools on `PATH` (and
only if you choose to compile the IR by hand), with no build-time dependency. The MIR→IR lowering
lives behind the shared `Backend` seam, so an inkwell implementation can be dropped in later without
touching the driver.

## Installing LLVM (Windows) — optional

You only need the command-line tools (`clang`, and optionally `opt`/`llc`) if you want to compile the
**emitted textual IR** yourself; the everyday `--run` / `--emit=obj|exe` paths do not use them.

1. Download `clang+llvm-19.1.7-x86_64-pc-windows-msvc.tar.xz` (or the plain `LLVM-19.1.7-win64.exe`
   installer) from the `llvmorg-19.1.7` release at <https://github.com/llvm/llvm-project/releases>.
2. Extract/install it, e.g. to `C:\llvm\19.1.7`.
3. Add its `bin` directory to `PATH` so `clang --version` works in a fresh shell.

Then, to hand-compile emitted IR (an optional external step — Cranelift's `--emit=exe` needs none of
this):

```sh
wukongc --emit=llvm-ir -O2 examples/fib.wk > fib.ll   # textual IR
clang -c -O2 fib.ll -o fib.o                            # compile with your LLVM tools
```

`fib.ll` is this module (as printed on x86-64 Windows; `crates/wukongc/tests/emit.rs` fails if it
drifts from what the compiler emits). At `-O2` the second recursive call has become the loop through
`%bb3`, and `print` is the runtime's `i64` printer:

<!-- emitted: wukongc --emit=llvm-ir -O2 examples/fib.wk -->
```llvm
; Wukong -> LLVM IR
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @wukong_rt_print_i64(i64)

define i32 @fib(i32 %v0) {
bb0:
  br label %bb3
bb1:
  %v19 = add i32 %v16, %v15
  ret i32 %v19
bb2:
  %v8 = sub i32 %v15, 1
  %v9 = call i32 @fib(i32 %v8)
  %v12 = sub i32 %v15, 2
  %v18 = add i32 %v16, %v9
  br label %bb3
bb3:
  %v15 = phi i32 [ %v0, %bb0 ], [ %v12, %bb2 ]
  %v16 = phi i32 [ 0, %bb0 ], [ %v18, %bb2 ]
  %v4 = icmp sge i32 %v15, 2
  br i1 %v4, label %bb2, label %bb1
}

define i32 @main() {
bb0:
  %v1 = call i32 @fib(i32 10)
  %t0 = sext i32 %v1 to i64
  call void @wukong_rt_print_i64(i64 %t0)
  ret i32 0
}
```

`fib.o` then links exactly like the object `--emit=obj` writes: its only external symbol is the
`wukong_rt_print_i64` shim (a program with recognized kernels also references `wukong_*` runtime
entries), which the `--emit=exe` link supplies (the rustc-driven shim plus the `wukong_runtime`
rlib, or the C runtime for a kernel-free program).

Beyond the module's own `declare`s, LLVM's code generator itself calls the C library for an
intrinsic the target CPU has no instruction for: at the baseline x86-64 level `llvm.roundeven`,
`llvm.floor`/`ceil`, `llvm.fma` and `half` conversions become calls to `roundevenf`, `floorf`/`ceilf`,
`fmaf` and `__truncsfhf2`/`__extendhfsf2`. Compile with `-march=x86-64-v3` (or at least
`-msse4.1 -mfma -mf16c`) to get instructions instead — the MSVC C runtime has no `roundevenf`.

If you later want the in-process inkwell path, standardize on
`inkwell = { version = "0.9", features = ["llvm19-1"] }` (note: LLVM 18+ use the `-1` feature
suffix), install the **MSVC** `clang+llvm` archive (it includes the dev libs + `llvm-config.exe`),
and set `LLVM_SYS_191_PREFIX` to its root. Keep the Rust toolchain on MSVC — do not mix it with the
MinGW/UCRT gcc, whose ABI does not match.

## The everyday native path (no LLVM)

`--train --backend=native` also uses the in-process Cranelift JIT and needs no LLVM. It executes
training immediately; `--train --emit=exe` (or `--emit=obj`, or `-o`) instead builds a standalone
training executable through the same Cranelift object path and the rustc link, with no LLVM either.
Any other explicit `--emit` stage with `--train` is rejected.

```sh
wukongc --emit=exe -O2 examples/fib.wk   # -> fib.exe, via Cranelift + the rustc/cc link
./fib.exe                                   # prints 55 (main returns 0)
```

`--emit=exe` also leaves the intermediate object `fib.o` beside the invocation — the same artifact
`--emit=obj` produces. The *generated link inputs* (the Rust shim, or the C runtime on the fallback
path) go into a temporary scratch directory keyed by pid and file stem and are deleted when the link
finishes, so they stay out of your working directory and two concurrent links of the same stem cannot
collide. Only if that scratch directory cannot be created does the driver fall back to the historical
behaviour of writing them beside the invocation — and on that path the generated `{stem}_rt.c` is left
behind (only the Rust shim is removed explicitly). Note `-o` is accepted only with `--emit=obj` or `--emit=exe`; every other stage prints its
artifact on stdout.
