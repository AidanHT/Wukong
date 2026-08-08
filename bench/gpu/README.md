# `bench/gpu/` — raw round logs from rented GPUs

Required by `GPU_RETARGET_PLAN.md` §6.5: *"Tune caches and raw round logs committed under
`bench/gpu/<device>/` as artifacts."* This directory is the evidence trail — every number that ever
appears in a doc must be traceable to a file here.

The operational checklist that produces these files is
[`docs/gpu/phase1-runbook.md`](../../docs/gpu/phase1-runbook.md).

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

1. **Every log opens with the §6.1 provenance block** (template in the runbook §2.2): local time,
   operator, checkout path, git branch, **git HEAD**, **git dirty state**, provider + SKU, image tag,
   the exact command, cargo profile, device identity (name / CC / SM count / opt-in SMEM / L2 / VRAM),
   driver + runtime, MIG state, the provenance-gate verdict, clock-lock status, wall time, exit code,
   and cost. A log without it is not evidence.
2. **HEAD and dirty state are recorded LOCALLY, before the command runs.** The Modal mount ships the
   **working tree, not a commit**, and excludes `.git` — so nothing on the remote side can identify
   the source. **A dirty tree on a metered run is a provenance violation:** its numbers cannot be
   reproduced and must not be published.
3. **A round whose device does not match spec publishes nothing** (plan §6.1). Keep the log — a
   MISMATCH is itself a finding — and say plainly in the summary that nothing from it is quotable.
4. **Container rounds are iteration data; VM rounds are publication data** (plan §6.3). Modal cannot
   lock clocks, so nothing measured there belongs in `BENCHMARKS.md`, `docs/metrics.md` or
   `docs/compile-floor.md`. Label every log with its clock-lock status.
5. **Wall times are first-class.** Suite and bench wall time is what sizes later metered phases
   (plan §6.6) — record it even when the run is "just" a correctness gate.
6. **Cost is recorded per round** (plan §6.6): the estimate *and* the provider dashboard's actual,
   with the discrepancy noted rather than reconciled away.
7. Strip ANSI escapes before committing (the image sets `CARGO_TERM_COLOR=always`); see the runbook
   §2.4 for the one-liner.
8. Logs are **append-only records**. Never edit one to look better. A correction is a new commit that
   states what was wrong — this project has published four such retractions and each one made the
   instrument stronger.
9. Stage explicitly (`git add bench/gpu/<device>/<file>`); **never `git add -A`**, and never add a
   `Co-Authored-By` trailer.
