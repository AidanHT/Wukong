# `bench/gpu/` — raw round logs from rented GPUs

**Measurement evidence, not the language.** These files are append-only logs from rented-GPU campaigns
(`tools/cloud/modal_app.py`, strong-peer bars for flash/conv/serving/GEMM). The language product
surface is user-written `.wk` in [`examples/`](../../examples/) and [`tests/run/`](../../tests/run/).

Tune caches and raw round logs are committed under `bench/gpu/<device>/` as artifacts. Every
number that appears in a doc must be traceable to a file here.

## Layout

```
bench/gpu/<device>/<YYYY-MM-DD>-s<N>-<step>.log     one raw log per command, in run order
bench/gpu/<device>/<YYYY-MM-DD>-session.md          provenance + cost ledger + findings + verdict
bench/gpu/<device>/autotune-<device>.json           persisted tune cache, when a phase produces one
```

`<device>` is the rented part, lowercase: `l4`, `l40s`, `h100`, `a100-40`, `sm120`. The dev laptop's
own runs stay where they already live — this tree is for **rented** silicon.

Re-running a step on the same date: suffix `-r2`, `-r3`. **Never overwrite a log**; a superseded
round is superseded by a *later* file plus a sentence in the session summary saying why.

## Rules

1. **Every log opens with the provenance block**: local time,
   operator, checkout path, git branch, **git HEAD**, **git dirty state**, provider + SKU, image tag,
   the exact command, cargo profile, device identity (name / CC / SM count / opt-in SMEM / L2 / VRAM),
   driver + runtime, MIG state, the provenance-gate verdict, clock-lock status, wall time, exit code,
   and cost. A log without it is not evidence.

   The device half of that block is now **emitted by the harness itself** —
   `wukong_codegen_gpu::bench_instrument`'s `Round::header()` prints device identity, the spec-table
   verdict, driver, provider/SKU, clock-lock status, clocks/temp/power **before and after**, drift,
   wall time and cost. A block printed by the code that took the measurements cannot be forgotten by
   whoever ran it; the shell-side items above (HEAD, dirty state, image tag) still come from the
   operator.

   **A round whose device does not verify against the spec table cannot open**, so it publishes
   nothing by construction rather than by discipline — but the refusing round still formats a
   provenance block (`Provenance::header()`), because rule 3 keeps the log.
2. **HEAD and dirty state are recorded LOCALLY, before the command runs.** The Modal mount ships the
   **working tree, not a commit**, and excludes `.git` — so nothing on the remote side can identify
   the source. **A dirty tree on a metered run is a provenance violation:** its numbers cannot be
   reproduced and must not be published.
3. **A round whose device does not match spec publishes nothing.** Keep the log — a
   MISMATCH is itself a finding — and say plainly in the summary that nothing from it is quotable.
4. **Container rounds are iteration data; VM rounds are publication data.** Modal cannot
   lock clocks, so nothing measured there belongs in `BENCHMARKS.md`, `docs/metrics.md` or
   `docs/compile-floor.md`. Label every log with its clock-lock status.
5. **Wall times are first-class.** Suite and bench wall time is what sizes later metered phases
   — record it even when the run is "just" a correctness gate.
6. **Cost is recorded per round**: the estimate *and* the provider dashboard's actual,
   with the discrepancy noted rather than reconciled away.
7. Strip ANSI escapes before committing (the image sets `CARGO_TERM_COLOR=always`).
8. Logs are **append-only records**. Never edit one to look better. A correction is a new commit that
   states what was wrong — this project has published four such retractions and each one made the
   instrument stronger.
9. Stage explicitly (`git add bench/gpu/<device>/<file>`); **never `git add -A`**, and never add a
   `Co-Authored-By` trailer.
