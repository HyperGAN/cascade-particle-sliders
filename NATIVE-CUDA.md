**Current source is [PG0.9.5](docs/pg095.md).** The PG0.9.4 native receipts and measured rates below are historical; they do not calibrate the new source.

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

Installed CPU verification of this source: 52 tests pass with CUDA hidden,
including exact continuation, native-only exports, signed checkpoint VJPs,
source/profile refusals and cleanup. The fresh Python 3.12 environment uses
Torch 2.9.1+cu128 through a selective read-only dependency site; no competing
product/core/PG distribution is exposed by that site. Exact core / HG VCS wheels
were reused from their original pip cache with unmodified public origins.
`pip check` passes. These checks do not qualify a real pretrained CUDA run.

The first fresh-process CUDA fixture correctly refused resume because the tiny
Diffusers constructor's unordered `_use_default_values` metadata differed across
isolated Python processes. That failed attempt is retained. V3 now supplies every
constructor option explicitly, leaving no implicit-default list; no old packet
is sorted, rewritten or rebound. A fresh source-bound CUDA proof is required.

Construction seeds the explicit CPU generator and, for the CUDA profile, only
the owned CUDA0 generator within its RNG-restoration scope. It never uses
`torch.manual_seed`, which would seed every visible accelerator. The regression
refuses CPU paths that call CUDA-wide seeders.

The final source passed a fresh-process tiny native CUDA software proof: actual
G/D updates, direct teacher/OFF parity, full two-update state and metrics equal
to one update plus fresh-process restore, immutable base, and raw inference at
strengths 0 / .5 / 1 without constructing a game or critic. Torch allocation peaked
at 0.033 GiB for this tiny random model under a 2 GiB role cap. The compact receipt
is [native-cuda-software-proof-20261104.json](docs/native-cuda-software-proof-20261104.json).
This result supplies software evidence for the profile. The separate pretrained
raw-response proof below has its own data, memory and qualification scope.

## Pretrained native256 raw-response proof

The same runtime `17e9f7b` passed a pretrained Stage C CUDA probe with 3.589 billion parameters using
the exact cached prior `621fc2d` and explicitly audited metadata view `7ca32c2`.
At native fields `[1,16,6,6]` (256 pixels), the frozen corpus contains 18 training,
18 development and 6 no-edit fields from stock 50-step DDPMWuerstchen / CFG 4
trajectories. Direct teacher/OFF parity, actual G/D updates, immutable base,
fresh-process two updates versus one-plus-one complete state/metrics, and
portable raw inference on 18 development fields at strengths 0 / .5 / 1 all pass.
Torch allocation peaked at 13.633 GiB under a 20 GiB admitted role. All owned
processes and role locks were released on completion.

The two-update diagnostic uses G = .0012, D = .018, no bank and a 64-update
recipe horizon. Development relative squared error is 1.001309 versus OFF = 1,
so it establishes execution rather than useful optimization. A production
LR recommendation, longer matched screens and complete native Stage B/A
image-quality comparisons remain pending. Search checkpoints do not add
trained/published product controls. See the compact exact-source/data receipt
[native-pretrained-response-proof-20261104.json](docs/native-pretrained-response-proof-20261104.json).


## Pretrained native1024 mechanical proof

The same runtime `17e9f7b` now passes the complete pretrained response proof at
`[1,16,24,24]` fields / 1024-pixel geometry. A new stock50-step CFG4 corpus has
18 training,18 development and6 no-edit fields; confirmation remains sealed.
Actual initial full packets match across fresh processes, actual G/D updates
change their leaves while the base stays exact, two uninterrupted updates equal
one plus a fresh-process restore, and portable raw responses match all18
held-out fields at strengths0/.5/1 without constructing training objects.
All five phases completed and the24GiB reservation was released. The largest
Torch allocation was15.478796GiB, with a23GiB allocator cap inside that total.

This two-update probe uses G`.0012`, absolute D`.018`, no bank, and a128-update
recipe horizon. Its equal-example pooled development relative SSE is`.9996815`
(OFF is1); this small diagnostic result does not recommend a learning rate.
A separate full native1024 sampler/StageB/A decode also produced recognizable
brass-robot OFF and pixel-art teacher images in100.005seconds, peaking at
16.354316GiB. Their scene/proportion changes do not establish preservation.

The frozen worker additionally checks highest FP32 precision, reduction flags,
TF32 environment overrides and the actual physical UUID. Those extra checks
are not all enforced by the narrower public v3 runtime validator. Portable
replay is verified under the exact frozen worker, not under arbitrary caller
settings. Its successful guardian stayed alive throughout; subsequent workers
must inherit physical/role lease FDs and compare all data bytes to original
capture receipts at each phase entry. The archived v1 source was not changed
in flight. See [the exact measured receipt](docs/native1024-pretrained-mechanics-20261104.json).
Runtime, install requirements and historical artifacts are unchanged.
