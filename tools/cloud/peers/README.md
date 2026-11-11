# `tools/cloud/peers/` — strong-peer scripts

Python helpers for resolving, verifying, and smoke-testing **library peers** (cuBLAS, CUTLASS,
FlashAttention, `torch.compile`, vLLM Marlin/Machete) on rented GPUs. These scripts are invoked by
[`modal_app.py`](../modal_app.py) — they are not Wukong programs.

| File | Role |
|------|------|
| `verify_peers.py` | Resolves every declared peer; exits non-zero if a required one is missing |
| `smoke_inductor.py` | Proves Inductor emitted a Triton kernel; warms autotune cache |
| `torch_compile_peer.py` | `torch.compile` framework bar (eager / compiled / max-autotune) |
| `selftest_modal_helpers.py` | Off-device tests for staging logic lifted from `modal_app.py` |

Run off-device selftest:

```sh
python tools/cloud/peers/selftest_modal_helpers.py
```
