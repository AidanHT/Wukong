# `bench/gpu/h100/` — H100 rented-GPU round logs

Raw logs from Modal/VM H100 campaigns: CPU staging (`*-build-cpu.log`), device suite passes
(`*-device-suite.log`, `*-full-suite.log`), wgmma bring-up, Act-2 peer bars (`wgmma_vs_cublas`,
cluster sweeps), and preflight `build_peers` rounds. Files are named
`<YYYY-MM-DD>-<step>.log` plus `*-session.md` summaries.

These document **what was measured on rented Hopper silicon** — strong-peer GEMM/attention bars,
not the Wukong language surface. Clock-locked VM rounds are publication data; Modal container rounds
are iteration data (see `tools/cloud/README.md`).
