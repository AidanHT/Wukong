# The optional LLVM path (`--emit=llvm-ir`)

**You do not need LLVM to build, run, or produce native binaries with Mercury.** The default native
path is **Cranelift** (pure Rust, in-process): `--run --backend=native` JITs, and `--emit=obj` /
`--emit=exe` write a native object / executable with **no LLVM toolchain** (verified on a box with no
`clang` installed — `--emit=obj` emits a COFF/ELF object directly from Cranelift). `--emit=exe` links
that object with the system C compiler (`cc` / `$CC`); if none is found it exits with code 2
(`UNIMPLEMENTED`) but still writes the object. To run with zero external toolchain at all, use the
interpreter: `mercuryc --run program.mer`.

LLVM enters **only** through `--emit=llvm-ir`, which prints **textual LLVM IR** for inspection — and
even that needs no LLVM installed (it is just text on stdout):

```sh
mercuryc --emit=llvm-ir -O2 examples/fib.mer    # print textual LLVM IR
```

The only reason to install LLVM is to take that emitted textual IR and compile it yourself with an
external `clang`/`llc`, or to build a future in-process `inkwell` backend. Neither is required for
`--run`, `--backend=native`, `--emit=obj`, or `--emit=exe`.

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
mercuryc --emit=llvm-ir -O2 examples/fib.mer > fib.ll   # textual IR
clang fib.ll ...                                        # compile with your LLVM tools
```

If you later want the in-process inkwell path, standardize on
`inkwell = { version = "0.9", features = ["llvm19-1"] }` (note: LLVM 18+ use the `-1` feature
suffix), install the **MSVC** `clang+llvm` archive (it includes the dev libs + `llvm-config.exe`),
and set `LLVM_SYS_191_PREFIX` to its root. Keep the Rust toolchain on MSVC — do not mix it with the
MinGW/UCRT gcc, whose ABI does not match.

## The everyday native path (no LLVM)

```sh
mercuryc --emit=exe -O2 examples/fib.mer   # -> fib.exe, via Cranelift + system cc
./fib.exe                                   # prints 55 (main returns 0)
```
