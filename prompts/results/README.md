# Per-branch results (keeps `BENCHMARKS.md` conflict-free)

Each parallel session writes its measured numbers, plan, and findings here in its **own** file
(`gemm-cliff.md`, `quant.md`, `attention.md`, `conv.md`, `cpu-library.md`, `serving.md`) instead of
editing `BENCHMARKS.md`/`CHANGELOG.md` directly — those are off-limits to the sessions and are
consolidated by the human at merge time. This is the single biggest cross-branch conflict source, so it
is fenced off here. Record only **same-run ratios / %-of-peer** (absolute TFLOP/s is meaningless — the
GPU clock swings ~7×, the CPU ~3×), and name every peer precisely.
