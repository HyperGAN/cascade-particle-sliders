# Cascade Stage C ParticleGAN 0.9.5 learning rate measurement

The best tested source-qualified recipe uses G `0.0012` and actual D `0.018000000000000002` (`d_lr_mult=15.000000000000004`), with no particle bank. The selected run completed 128 updates at horizon128.

This is a native1024 Stage C prior response measurement with FP32 base, adapters and responses, rank4, batch2, and gradient checkpointing. It does not qualify Stage B/A decoding or rendered image quality. The fixed-D G screen retained every actual available128 endpoint; this is not a joint or global optimum.

The independent final RAW relative SSE is **1.0226831513985883** (OFF1, teacher0, reversed edit4), from 6 teacher-defined examples and 0 exact-zero exclusions. Teacher-only membership was fixed before the student. All student values, including excluded examples, were finite. Final values above OFF are retained and never trigger tuning.

The selected checkpoint and RAW artifact were frozen before the first semantic cohort read. An earlier final attempt stopped at schema validation before loading a model or producing teacher/student responses or a score. The repair authenticated the same original cohort bytes and ordered seed list; no subjects, seeds, checkpoint, rate or metric were changed, and the final was not used for tuning.

The score casts operands to FP64 before subtraction, pools all coordinates at positions5/25/45 within each example, and averages examples equally. An independent CPU audit exactly reaggregated the stored native CUDA field scalars. Its separate CPU tensor score `1.0226831513985886` passed the declared FP64 accumulation bound; it does not replace the native score. Native checkpoint/RAW-network and public replay checks remain native-qualified assertions.

For a fresh run, merge `configs/lr/pg095-native1024-h128.options.json` with your supported staged model directory and source-qualified native teacher cache. Set `ema_decay=None` in Python (`null` in JSON), and export/evaluate/infer with `selection="raw"`. Weight EMA is disabled for model, adapter, critic and bank; optimizer scalar statistics are unchanged. Defaults are unchanged.

This is a fresh-training overlay. Resume an existing checkpoint using its literal authenticated SessionConfig: historical packets can omit defaultNone/rank4 keys, so adding explicit keys must not silently rewrite resume identity. See `docs/lr-evidence/cascade-pg095-native1024-fp32-128.json` for sanitized evidence.
