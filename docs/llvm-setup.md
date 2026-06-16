# Native code generation with LLVM

Mercury's LLVM backend emits **textual LLVM IR** (`--emit=llvm-ir`) and then compiles it to a
native object or executable with `clang` (`--emit=obj` / `--emit=exe`). This keeps the default
build dependency-free: you only need LLVM tools installed when you want native binaries. To run a
program without any LLVM install, use the interpreter: `mercuryc --run program.mer`.

## Why textual IR (and not inkwell in-process)

The original plan targeted `inkwell` (the in-process libLLVM API). On Windows that path is
fragile: `llvm-sys` requires the LLVM **development** libraries plus `llvm-config`, which the stock
`LLVM-*-win64.exe` installer does **not** ship, and the Rust (MSVC) / LLVM build ABIs must match.
Emitting textual IR sidesteps all of that — it needs only the LLVM command-line tools on `PATH` and
no build-time dependency. The MIR→IR lowering lives behind the shared `Backend` seam, so an inkwell
implementation can be dropped in later without touching the driver.

## Installing LLVM (Windows)

You only need the command-line tools (`clang`, and optionally `opt`/`llc`):

1. Download `clang+llvm-19.1.7-x86_64-pc-windows-msvc.tar.xz` (or the plain `LLVM-19.1.7-win64.exe`
   installer) from the `llvmorg-19.1.7` release at <https://github.com/llvm/llvm-project/releases>.
2. Extract/install it, e.g. to `C:\llvm\19.1.7`.
3. Add its `bin` directory to `PATH` so `clang --version` works in a fresh shell.

Then:

```sh
mercuryc --emit=exe -O2 examples/fib.mer   # -> fib.exe
./fib.exe; echo $?                          # 55
```

If you later want the in-process inkwell path, standardize on
`inkwell = { version = "0.9", features = ["llvm19-1"] }` (note: LLVM 18+ use the `-1` feature
suffix), install the **MSVC** `clang+llvm` archive (it includes the dev libs + `llvm-config.exe`),
and set `LLVM_SYS_191_PREFIX` to its root. Keep the Rust toolchain on MSVC — do not mix it with the
MinGW/UCRT gcc, whose ABI does not match.

## Inspecting IR

```sh
mercuryc --emit=llvm-ir -O2 examples/fib.mer    # print textual LLVM IR
```
