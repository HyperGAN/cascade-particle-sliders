# Explicit learning rates

Direct preparation and common session API/JSON configuration accept `lr`,
`d_lr_mult`, and `prior_lr_mult`. Omitted overrides preserve ParticleGAN 0.9.4
base 0.012 and critic multiplier 1.5. Explicit base and critic values must be
finite and positive. Any non-null prior multiplier is refused because Cascade
is adapter-only and has no trainable particle bank.

The common CLI consumes the same JSON options; there is no second LR default.

The shared game constructs actual optimizer groups and binds the resolved
recipe into exact resumable identity. Changed rates require a new run.
Historical source-bound packets and receipts keep their original identities;
no rebinding or old source-lock rewrite is provided.

Installed tiny-native CPU tests cover actual optimizer rates and resume.
The existing native admission, frozen-teacher, precision and rendering gates
remain in force. These controls do not establish tuned native learning rates.
