# `bench/gpu/l4/` — L4 rented-GPU round logs

Raw logs from the cheapest `sm_89` correctness box on Modal: `device_info`, debug/release builds,
peer smoke tests (cuBLAS, cuDNN, cuBLASLt fp8), and early suite passes. Same architecture family as
the dev RTX 4050 — validates Linux/`cudarc`/driver-JIT before renting Hopper.

Append-only measurement records with `*-session.md` provenance ledgers. Not language fixtures —
see [`tests/run/`](../../tests/run/) for `.wk` correctness programs.
