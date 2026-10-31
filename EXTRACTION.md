# Standalone Cascade source export

Source: `HyperGAN/particle-sliders` commit
`11c51752179f24b7742545bee0541b2799ffb8b2`, product root
`conceptmod/cascade-shared/`. Destination preparation:
`HyperGAN/cascade-particle-sliders`.

This is a clean allowlisted source export. No history is grafted or mirrored.
`PROVENANCE.json` maps each original repository path and Git blob to the new
root-relative path, with original/exported SHA256 and any adjustment. Runtime
Python, import names, HNDL graph, shared-core closure lock, tests and tiny config
remain byte-identical. README installation/test paths are adapted to this root;
package metadata now includes the original repository MIT license verbatim.

The imported paths are the nine product files listed in that manifest plus the
original root `LICENSE`. Generated extraction documents are `PROVENANCE.json`
and this file. No historical trainer/model-specific file is imported: the
historical `conceptmod/textsliders/train_lora_cascade.py` family remains in the
original repository, with its objectives/default artifacts intact.

Private validation documents, receipts, logs, environment paths, secrets,
recordings, weights and run outputs are excluded. There are no workflows,
deployment files, release actions or permission changes. Only minimal review
scaffolding belongs on the new default branch; actual product source belongs on
a separate draft review branch until root completes its review/admission gates.

The package retains namespace `particle_sliders_cascade` and both
`cascade_stage_c` plugin entry points. Its shared dependency remains exactly
`particle-sliders-core[hypergan-hndl]` at
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, including existing generic
`particle_sliders.fibo_hndl` compatibility imports. PG 0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL 0.10.0 are unchanged.
No shared mathematics, API, network optimization or domain policy is rewritten.

Pure source relocation should preserve the versioned factory/adapter identities
because their module names and source/graph/lock bytes remain identical. A
configuration, frozen-base path/content, source revision, precision or dependency
change can still invalidate checkpoint/artifact identity. No automatic rebind
or migration is introduced. Cross-install continuation and native inference
must be checked with installed distributions and both checkouts absent from
`sys.path`; fixture results do not qualify full pretrained/native models.

One existing limitation surfaced during extraction checks: the constructor-only
tiny fixture can carry a differently ordered Diffusers `_use_default_values`
list in another process, which strict identity checks correctly refuse. The
export does not normalize or rebind that metadata. Cross-install continuation
and native inference are checked with a fully specified local tiny-prior config;
the runtime, model code and identities themselves remain unchanged.

Protocol v2 preserves actual Stage C ratio/conditioning and numerical response
geometry. Critic time uses the explicit unit-ratio feature view. Public strength
zero is OFF; raw-host default remains +1 and finite signed extrapolation remains
software support. EMA/resume/native-only inference and rejection/cleanup/RNG
contracts remain the reviewed product contracts. Entirely zero-guidance training
is rejected; mixed zero-guidance rows remain frozen-negative controls.

Core admission and centralized GPU/native qualification remain pending. No GPU
job, full-model training, release, deployment or merge is authorized by this
source extraction. The original embedded paths remain until replacement is
verified. Root owns Claude actual-source/RETURN review and final decisions.
