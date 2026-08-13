# `tools/perf/` — local performance helpers

Scripts for measuring compiler and kernel performance on the dev machine. These drive **Rust benches
and peers**, not user-authored `.wk` programs — see [`examples/`](../examples/) and
[`tests/run/`](../tests/run/) for the language surface.

Contents vary by campaign; check the directory for current scripts. Results that become publication
evidence land under [`bench/gpu/`](../bench/gpu/) with provenance logs, not here.
