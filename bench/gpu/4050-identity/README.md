# `bench/gpu/4050-identity/` — dev laptop identity logs

Raw round logs from the local RTX 4050 (`sm_89`): device-info probes, cubin-cache compile-latency
A/B rounds (`cubin_cache_compile_latency.*.txt`), and bring-up sessions. These establish **device
identity** (CC, SM count, L2, opt-in SMEM) for the dev instrument — not language correctness
fixtures.

Session summaries (`*-session.md`) record provenance, cost, and verdicts. Never overwrite a log;
suffix `-r2`, `-r3` for reruns.
