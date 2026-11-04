# Current shared-core cohort

The active package 0.1.1 pins `particle-sliders-core[hypergan-hndl]` 0.12.0 to
`HyperGAN/particle-sliders@ddb41423bcefc1fe76cd94d0f9b367353b3ea137`.
This includes the merged shared training dispatcher and isolated execution worker.
ParticleGAN is the official 0.9.4 release, source
`330eb94ee748cdcad131b0275d8de75fd5a7f96f`; the core checks its installed release
source manifest. HyperGAN remains `5f368a3770cf5bdae0448995c9f76e29af8049f7`
and HNDL remains 0.10.0.

`core-lock.json` v2 covers all **574** installed source/data files: 573 in
`particle_sliders` and one in `concept_slider_core`. Every hash was checked
against the exact Git tree. Admission checks distribution version, exact VCS
origin/commit, import shadowing and the recursive installed bytes, then the
official PG release source. Missing, modified, extra and symlinked source/data
refuse before native allocation or adapter mutation. The distribution's
RECORD-tracked wheel-root `.pth` hook is separate from these two package trees.

This is a new source/closure identity. No old checkpoint, artifact or cache
metadata is relabeled. The exact a794 lock, config, provenance and evidence are
retained under `docs/archive/a794/`; earlier 4abc and original extraction
lineage remain recorded there and in `PROVENANCE.json`. Use the original pinned
package/environment to reproduce an older cohort. Current training requires a
fresh cohort/output/cache identity.

The adapter-only generator arm is unchanged: `particles=None`, no particle bank,
no SBNP bank regularization, no substitute local GAN formula. Shared PG owns the
logistic objective, BISPF critic, optimizer law and update lifecycle. This is a
supported adapter GAN arm rather than the complete bank-based ParticleGAN recipe.
Stage C numerical embedding responses remain the only product surface; no
Stage B/A trainer or renderer is added.

## CPU install and checks

Use a separate Python 3.11–3.14 environment. The verified environment was Python
3.12.15 / Torch 2.9.1+cpu, installed from public package/VCS requirements with no
system site-packages. `pip check` passes. No pretrained model is downloaded.

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install build pytest .
.venv/bin/python -m build --outdir dist
.venv/bin/python -m pip install --force-reinstall --no-deps dist/*.whl
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -I -m pytest --noconftest -c /dev/null --import-mode=importlib tests -q
.venv/bin/python -m pip install --force-reinstall --no-deps dist/*.tar.gz
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -I -m pytest --noconftest -c /dev/null --import-mode=importlib tests -q
.venv/bin/python -m pip check
```

Installed wheel and sdist each pass **33 tests**, outside the source import roots.
These cover HNDL outputs/VJPs/second derivatives, actual adapter updates,
zero/signed strengths, EMA/native-only API inference, exact resume, rollback,
cleanup retry, PG/source admission and both namespaces' closure rejection.
The dispatcher test plans, checks, rejects `dummy=false` for the fixture profile,
trains two of four updates with periodic checkpoints, resumes to four, matches
an uninterrupted worker checkpoint exactly and exports both v1 and EMA v2.

The supplied `tiny-profile.json` declares `status=fixture` and requires the actual
`dummy=true` config. It fixes `PYTHONHASHSEED=0` for worker conformance. The existing
constructor-only Diffusers `_use_default_values` ordering limitation still makes
some cross-import-path/cross-process identities differ and correctly refuse;
this change does not normalize or rebind that identity. Previously documented
fully specified staged tiny-prior checks are distinct from this fixture check.
No additional native model or GPU qualification is claimed.

Actual-source Claude RETURN/root review, full pretrained Stage C qualification,
DDPMWuerstchen sampling plus Stage B/A rendering parity and native GPU/model
quality remain pending. No model weights, GPU training, release or merge is
included.


## V3 explicit execution profile

The source update from277ec1c retains the same core/PG/HG/HNDL locks and
adapter-only mathematics. It binds CPUdefault or explicit strictFP32 CUDA
execution, numerical flags and checkpointing choice in the newv3 native identity.
The oldv2 implementation and receipts are retained byte-for-byte under
`docs/archive/stage-c-v2/`; use its pinned Git source for old artifacts.
See [NATIVE-CUDA.md](NATIVE-CUDA.md) for required flags and qualification gates.
Installed CPU and native CUDA results for this update must be recorded separately
from the older33-test source cohort above.
