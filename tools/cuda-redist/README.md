# `tools/cuda-redist/` — vendored CUDA redistributables

Vendored NVIDIA CUDA redistributable packages (cuBLAS, cuDNN, NVRTC, NVJITLink, etc.) used to build
and link **peer binaries** on machines without a full CUDA toolkit install. This tree is packaging
for `tools/cloud/peers/` and local peer smoke tests — not part of the Wukong language or the
`.wk` corpus.

When present, wheels and shared libraries live under `nvidia/` and `*.dist-info/` metadata trees.
Those paths are not documented here; consume them through the peer build scripts that reference this
prefix.
