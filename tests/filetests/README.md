# tests/filetests — what the compiler emits, checked line by line

FileCheck-style tests, run by `crates/wukongc/tests/filetests.rs`
(`cargo test -p wukongc --test filetests`; `WUKONG_FILETESTS=licm` runs the files whose path
contains `licm`). Each `.mir` or `.wk` file is a program plus, in its comments, the `RUN:` lines
that compile it and the `CHECK:` lines its output must match:

```text
; RUN: wukongc --emit=mir --passes=licm %s | filecheck %s
; CHECK-LABEL: fn invariant_product(
; CHECK:       bb0(
; CHECK:         [[AB:v[0-9]+]] = fmul v2, v3
; CHECK:       bb2:
; CHECK-NOT:     fmul v2, v3
```

| directory | what it pins |
|---|---|
| `opt/` | one optimizer pass per file: its essential transformation and a case it must leave alone. Mostly a hand-written `.mir` run through that pass alone, so the front end cannot drift under it; every file fails with its pass taken out of its `RUN:` line |
| `mir-text/` | the typed dump (`--emit=mir-text`) reads back to itself and is a faithful reproducer; `.mir` parse errors (their inputs are in `Inputs/`) |
| `verify/` | the MIR verifier rejects invalid `.mir` input, and `--verify-each -O2` changes nothing |
| `cli/` | `--print-after`, and pass lists as executable spellings of `-O0`/`-O1` |

This is the one corpus directory scanned **recursively**; files under an `Inputs/` directory are
read by `RUN:` lines and never run as tests. How to write one — the commands, the directives, and
the rule that expectations come from the pass's semantics and are confirmed, never pasted from
output — is in `CONTRIBUTING.md`, "Writing a filetest".
