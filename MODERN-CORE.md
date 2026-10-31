# Active modern core cohort

The active candidate pins `particle-sliders-core[hypergan-hndl]` to immutable
`HyperGAN/particle-sliders@a794e3b4a8813b012f2c4faa3da82b78d4a90214`.
The recursive lock covers all 50 installed source/data files. Its SHA256 is
`9332a438a90d1888a24de0f757e753342f9830ab35f89a35e0009ef5033a4f36`.
Core version remains 0.12.0, PG remains 0.9.4, HNDL remains 0.10.0 and HyperGAN
remains `5f368a3770cf5bdae0448995c9f76e29af8049f7`. Version alone is insufficient;
the exact Git origin and recursive installed bytes were verified.

This intentionally changes the product source/closure identity from the old
4abc preview cohort. Genuine old training state is rejected without changing a
live session, which remains usable for a subsequent update. Genuine old EMA
native-inference artifacts are refused before model allocation. Old metadata
is not rewritten, and old packets remain with their original loaders/cohorts.

Reproduce in an isolated CPU environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install build pytest .
.venv/bin/python -m build --outdir dist
.venv/bin/python -m pip install --force-reinstall --no-deps dist/*.whl
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -I -m pytest --noconftest -c /dev/null --import-mode=importlib tests -q
.venv/bin/python -m pip install --force-reinstall --no-deps dist/*.tar.gz
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -I -m pytest --noconftest -c /dev/null --import-mode=importlib tests -q
.venv/bin/python -m pip check
```

On CPU PyTorch 2.9.0+cpu / Python 3.14, installed wheel and sdist each passed all
29 product tests with source and umbrella checkouts absent from import roots;
pip check passed. Clean public dependency resolution selected the exact a794
Git origin and CPU wheel, without NVIDIA packages. Tests retain actual updates,
matched HNDL outputs/VJPs/critic second derivatives, OFF/raw/EMA/resume,
source-lock tamper refusal, explicit-primary cleanup/retry and RNG checks.

No product objective, bank, native geometry, scale policy, model license,
historical/default cohort, workflow or access policy changed. Shared-core
synthetic CUDA qualification does not qualify this model's native/GPU execution
or quality. Stage B/A, full pretrained/sampler/rendering and the existing
constructor-only fixture metadata-order limitation remain unqualified. No GPU
jobs, releases, model weights, deployment or merge are part of this change.
