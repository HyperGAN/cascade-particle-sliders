# Explicit Stage C CUDA response cohort

Protocol `cascade-stage-c-spatial-response-hndl-v3` introduces a separate
`native-cuda-fp32-math-v1` execution profile. The default remains
`cpu-fp32-v1` / `device="cpu"`. Old v2 exports and checkpoints require their
original source at `277ec1c2ec12fd4d97e2fd28090561d385aa445f`; exact files and
hashes are retained in `docs/archive/stage-c-v2/`. Do not rebind their metadata.

The CUDA profile requires `device="cuda:0"`, `base_precision="float32"` and
caller-controlled GPU admission. Before importing Torch, pin a single physical
GPU with `CUDA_VISIBLE_DEVICES` and set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Then set these flags before loading the model or capturing targets:

```python
import torch

torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

The package checks these flags before allocation and every native forward;
it does not change process-wide settings. The selected profile, flags, Torch
and CUDA versions, and checkpointing choice participate in native identity.
Portable inference retains that execution profile and refuses changed flags
before loading the base. A fresh profile requires freshly captured targets.

Frozen cache rows remain canonical CPU FP32 tensors. Native forward calls copy
latents, ratios and conditions to the selected device without altering those
rows. HNDL adapter leaves and critic are placed on that same device. The actual
Stage C output remains a spatial field, with unit-ratio time features and the
existing conditional-only CFG law. The adapter-only game has no particle bank.
PG remains the exact official 0.9.4 release and the shared core remains `ddb4142`.

Optional `gradient_checkpointing=True` uses the model's public enable API and
the shared core's non-reentrant, per-forward scale-capturing wrapper. Closing
the binding restores the borrowed model's checkpoint flags/functions, modes,
gradient objects and hooks. This is necessary for signed forwards whose
backward recomputation occurs after the scale context has unwound.

This source support is not a pretrained GPU or learning-rate qualification.
Before a rate screen, the native harness must verify exact direct teacher/CFG
parity, OFF identity, finite actual adapter/critic updates, full base immutability,
fresh-process uninterrupted versus resumed state, and portable raw inference.
Use captured identical initial tensors/data/RNG and separate development and
untouched confirmation examples. Common controls are `lr` and `d_lr_mult`;
`prior_lr_mult` is rejected because this product has no bank.

The package exposes numerical Stage C responses. Full native image comparison
additionally needs a source-bound DDPMWuerstchen sampler and Stage B/A decoder
parity; a raw embedding-field score does not establish image quality. Native
memory and throughput must be measured before assigning a GPU budget.

Installed CPU verification of this source: 51 tests pass with CUDA hidden,
including exact continuation, native-only exports, signed checkpoint VJPs,
source/profile refusals and cleanup. The fresh Python3.12 environment uses
Torch2.9.1+cu128 through a selective read-only dependency site; no competing
product/core/PG distribution is exposed by that site. Exact core/HG VCS wheels
were reused from their original pip cache with unmodified public origins.
`pip check` passes. These checks do not qualify a real pretrained CUDA run.
