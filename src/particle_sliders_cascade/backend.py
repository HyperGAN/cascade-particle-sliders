"""Cascade Stage C native geometry; shared core owns every GAN update.

This is a new positive-teacher adapter GAN cohort, not a migration of the
historical prompt_pair.loss objective. Stage B and Stage A are never trained.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from importlib.metadata import distribution, version
from pathlib import Path

import torch
from torch import nn

from particle_sliders.fibo_hndl import verified_builder, HNDLFieldCritic
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, Capabilities
from particle_sliders.pg09 import ConditionalPG09, PG_VERSION, PG_SOURCE_PROFILE, _hash, _identity
from particle_sliders.pg09_backends import AdapterParameters, FrozenResponseAdapter, ResponseRow
from particle_sliders.pg09_images import image_condition_features
from particle_sliders.response import ResponseBackend, ResponseCapabilities

NAME = 'cascade_stage_c'
PROTOCOL = 'cascade-stage-c-spatial-response-hndl-v2'
GRAPH = Path(__file__).with_name('lora.hndl')
CORE_COMMIT = 'ddb41423bcefc1fe76cd94d0f9b367353b3ea137'
CORE_LOCK = Path(__file__).with_name('core-lock.json')
FEATURE_POLICY = dict(version='cascade-unit-ratio-critic-v2', time_coordinate='native ratio r in [0,1]',
                      shared_helper_view='timestep=1000*r for features only; native row unchanged')
ADAPTER_POLICY = dict(version='cascade-explicit-activation-v2', raw_host_default=1.,
                      public_default_strength=1., strength_domain='all finite signed values',
                      strength_qualification='software semantics only; extrapolation quality unqualified')


def core_root():
    import particle_sliders
    return Path(particle_sliders.__file__).resolve().parent


def verify_core_closure():
    """Lock every installed recursive source/data byte to the pinned Git tree."""
    expected = json.loads(CORE_LOCK.read_text())
    if (expected['schema'], expected['commit'], expected['distribution'], expected['version']) != (
            'cascade-shared-core-closure-v2', CORE_COMMIT, 'particle-sliders-core', '0.12.0'):
        raise ValueError('Cascade core closure manifest contradicts the cohort')
    if version('particle-sliders-core') != expected['version']:
        raise ValueError('Cascade requires the locked shared-core distribution')
    import particle_sliders
    dependency = distribution('particle-sliders-core')
    origin = json.loads(dependency.read_text('direct_url.json') or '{}')
    if (origin.get('url') != 'https://github.com/HyperGAN/particle-sliders.git'
            or origin.get('vcs_info', {}).get('commit_id') != CORE_COMMIT):
        raise ValueError('Cascade requires the exact pinned shared-core Git origin')
    if Path(particle_sliders.__file__).resolve() != Path(dependency.locate_file('particle_sliders/__init__.py')).resolve():
        raise ValueError('Cascade shared-core import is shadowed')
    root = core_root().parent
    actual = {}
    for package in ('particle_sliders', 'concept_slider_core'):
        package_root = root / package
        if package_root.is_symlink() or not package_root.is_dir():
            raise ValueError('Cascade recursive shared-core source/data closure requires both packages')
        for path in package_root.rglob('*'):
            relative = path.relative_to(root)
            if '__pycache__' in relative.parts:
                continue  # Interpreter-generated cache is not packaged source/data.
            if path.is_symlink():
                raise ValueError('Cascade shared-core closure rejects symlinks')
            if path.is_file():
                actual[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected['files']:
        raise ValueError('Cascade recursive shared-core source/data closure differs before native mutation')
    # Admit the installed official PG release source before native mutation.
    from particle_sliders.pg09 import _released_sources
    _released_sources(profile=PG_SOURCE_PROFILE)
    return hashlib.sha256(CORE_LOCK.read_bytes()).hexdigest()


def unit_ratio_condition_features(prepared):
    """Adapt only the helper's time-feature view, never a native model row."""
    views = [replace(item, row=replace(item.row, timestep=item.row.timestep * 1000.)) for item in prepared]
    return image_condition_features(views)


def validate(options):
    verify_core_closure()
    allowed = {'dummy', 'model_dir', 'cache_file', 'cache_sha256', 'shape', 'rank',
               'steps', 'batch', 'seed', 'ema_decay', 'base_precision'}
    if set(options) - allowed:
        raise ValueError('unknown Cascade options: ' + str(sorted(set(options) - allowed)))
    shape = options.get('shape', [1, 4, 4, 4])
    if (not isinstance(shape, list) or len(shape) != 4 or shape[0] != 1
            or any(type(n) is not int or n < 1 for n in shape)):
        raise ValueError('Stage C requires explicit native [1,C,H,W] shape')
    if type(options.get('dummy', False)) is not bool:
        raise ValueError('dummy must be boolean')
    if options.get('base_precision', 'float32') != 'float32':
        raise ValueError('only CPU FP32 Stage C is qualified by this cohort')
    if type(options.get('rank', 4)) is not int or not 1 <= options.get('rank', 4) <= 128:
        raise ValueError('rank must be in [1,128]')
    if options.get('dummy', False):
        if shape != [1, 4, 4, 4] or any(k in options for k in ('model_dir', 'cache_file', 'cache_sha256')):
            raise ValueError('dummy fixture has fixed geometry and no staged paths')
    else:
        if not Path(options.get('model_dir', '')).is_dir():
            raise ValueError('stage the local Stage C prior directory first')
        path = Path(options.get('cache_file', ''))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != options.get('cache_sha256'):
            raise ValueError('frozen cache SHA256 differs before native allocation')
    if version('diffusers') != '0.36.0':
        raise ValueError('Cascade requires diffusers==0.36.0')
    verified_builder()  # Check HG/HNDL bytes before native mutation.


def capabilities(options):
    return ResponseCapabilities('adapter_gan', 'image', 'stage_c_image_embedding_prediction',
        'continuous_field', tuple(options.get('shape', [1, 4, 4, 4])), PG_VERSION, PG_SOURCE_PROFILE,
        strength_min=None, strength_max=None,
        qualification='tiny random native Stage C CPU conformance; pretrained and full Cascade rendering unqualified')


def tiny_model():
    from diffusers import StableCascadeUNet
    with torch.random.fork_rng():
        torch.manual_seed(51)
        return StableCascadeUNet(in_channels=4, out_channels=4,
            timestep_ratio_embedding_dim=8, conditioning_dim=8, block_out_channels=(16,),
            num_attention_heads=(2,), down_num_layers_per_block=(1,), up_num_layers_per_block=(1,),
            down_blocks_repeat_mappers=(1,), up_blocks_repeat_mappers=(1,),
            block_types_per_layer=(('SDCascadeResBlock', 'SDCascadeTimestepBlock', 'SDCascadeAttnBlock'),),
            clip_text_in_channels=8, clip_text_pooled_in_channels=8, clip_image_in_channels=8,
            clip_seq=2, dropout=0., timestep_conditioning_type=())


def fixture_rows():
    generator = torch.Generator().manual_seed(61)
    def condition():
        return dict(embeds=torch.randn(1, 3, 8, generator=generator),
                    pooled=torch.randn(1, 1, 8, generator=generator),
                    image_embeds=torch.zeros(1, 1, 8))
    return [ResponseRow(condition(), condition(), condition(),
            torch.randn(1, 4, 4, 4, generator=generator), torch.tensor(.25 + i * .25)) for i in range(2)]


def load_model(options):
    if options.get('dummy', False):
        return tiny_model()
    from diffusers import StableCascadeUNet
    return StableCascadeUNet.from_pretrained(options['model_dir'], local_files_only=True,
                                           torch_dtype=torch.float32, use_safetensors=True)


def model_identity(model):
    return json.loads(json.dumps(dict(stage='C', state_sha256=_hash(_identity(model.state_dict())),
                configuration=dict(model.config), diffusers=version('diffusers'),
                protocol=PROTOCOL, critic_feature_policy=FEATURE_POLICY, adapter_policy=ADAPTER_POLICY,
                shared_core_lock_sha256=verify_core_closure(),
                graph_sha256=hashlib.sha256(GRAPH.read_bytes()).hexdigest(),
                graph_runtime=verified_builder()[1], decoder_stages='B/A outside response cohort'), allow_nan=False))


def check_row(row, model, shape):
    if (not isinstance(row, ResponseRow) or row.schedule is not None or tuple(row.latent.shape) != tuple(shape)
            or row.latent.device.type != 'cpu' or row.latent.dtype != torch.float32
            or not torch.isfinite(row.latent).all() or row.scale != 1 or row.strength != 1
            or row.timestep is None or row.timestep.numel() != 1
            or not torch.isfinite(row.timestep).all() or not 0 <= float(row.timestep) <= 1
            or not 0 <= row.guidance < float('inf')):
        raise ValueError('finite CPU Stage C raw response row required; no flow trajectory')
    config = model.config
    if config.in_channels != shape[1] or config.out_channels != shape[1] or config.effnet_in_channels is not None:
        raise ValueError('requires Stage C prior, not Stage B/effnet decoder')
    expected = {'embeds': config.clip_text_in_channels, 'pooled': config.clip_text_pooled_in_channels,
                'image_embeds': config.clip_image_in_channels}
    for condition in (row.neutral, row.positive, row.negative):
        if not isinstance(condition, dict) or set(condition) != set(expected):
            raise ValueError('all native Stage C text/pooled/image conditions required')
        for key, width in expected.items():
            value = condition[key]
            if (not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1
                    or value.shape[-1] != width or value.device.type != 'cpu'
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError('invalid native Stage C conditioning: ' + key)


class LoRAGraph(nn.Module):
    def __init__(self, module, rank):
        super().__init__()
        builder, identity = verified_builder()
        self.graph = builder(GRAPH.read_text(), input_shape={'value': ['B', module.in_features]},
            output_shape=['B', module.out_features], parameters=dict(rank=rank, output_width=module.out_features))
        nn.init.kaiming_uniform_(self.graph['down'].weight, a=5 ** .5)
        nn.init.zeros_(self.graph['up'].weight)
        self.identity = dict(identity, graph_sha256=hashlib.sha256(GRAPH.read_bytes()).hexdigest())
        self.multiplier = ADAPTER_POLICY['raw_host_default']

    def extra_repr(self):
        return str(sorted(self.identity.items()))

    def delta(self, value):
        shape = value.shape
        return self.graph(value=value.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)


class NativeResponse(FrozenResponseAdapter):
    family = NAME
    time_coordinates = 'native DDPMWuerstchen timestep_ratio in [0,1]; Stage C embedding field'

    def __init__(self, model, network, identity):
        self.model = model
        super().__init__(network, model_identity=identity)

    def _velocity(self, condition, latent, timestep):
        return self.model(sample=latent, timestep_ratio=timestep.reshape(1),
            clip_text=condition['embeds'], clip_text_pooled=condition['pooled'],
            clip_img=condition['image_embeds']).sample.float()

    def metadata(self):
        return dict(super().metadata(), protocol=PROTOCOL, critic_feature_policy=dict(FEATURE_POLICY),
                    adapter_policy=dict(ADAPTER_POLICY), shared_core_lock_sha256=verify_core_closure(),
                    training_guidance_policy='reject all-zero objective; mixed zero rows are frozen-negative CFG controls')


class Cleanup:
    """Attempt every restoration, retain failed actions for an explicit retry."""
    def __init__(self):
        self.actions = []
        self.errors = []

    @property
    def pending(self):
        return bool(self.actions)

    def add(self, label, action, *, first=False):
        self.actions.insert(0, (label, action)) if first else self.actions.append((label, action))

    def __call__(self, *, primary=None):
        failed, errors = [], []
        for label, action in self.actions:
            try:
                action()
            except BaseException as error:
                failed.append((label, action))
                errors.append(error)
        self.actions, self.errors = failed, errors
        if errors:
            if primary is not None:
                primary.cascade_cleanup = self
                primary.add_note('Cascade cleanup incomplete; retry error.cascade_cleanup() or binding.close(): ' +
                                  '; '.join(label for label, _ in failed))
            else:
                raise BaseExceptionGroup('Cascade cleanup incomplete; retry the same cleanup callback', errors)


def attach(model, rank):
    """Transactional hooks: restore modes, gradients, buffers and checkpoint settings."""
    verify_core_closure()
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and name.endswith(('to_q', 'to_k', 'to_v', 'to_out.0'))]
    if not targets:
        raise ValueError('native Stage C attention projections missing')
    close, graphs = Cleanup(), nn.ModuleDict()
    def restore_value(tensor, saved):
        with torch.no_grad():
            if (tensor.shape, tensor.dtype, tensor.device) == (saved.shape, saved.dtype, saved.device):
                tensor.copy_(saved)
            else:
                tensor.data = saved.clone()
    for name, module in model.named_modules():
        close.add(name + ':mode', lambda module=module, mode=module.training: setattr(module, 'training', mode))
        for key in ('gradient_checkpointing', '_gradient_checkpointing_func'):
            if hasattr(module, key):
                close.add(name + ':' + key, lambda module=module, key=key, value=getattr(module, key): setattr(module, key, value))
            else:
                def remove_attribute(module=module, key=key):
                    if hasattr(module, key):
                        delattr(module, key)
                close.add(name + ':' + key, remove_attribute)
        def registrations(module=module, buffers=dict(module._buffers), nonpersistent=set(module._non_persistent_buffers_set)):
            module._buffers.clear()
            module._buffers.update(buffers)
            module._non_persistent_buffers_set.clear()
            module._non_persistent_buffers_set.update(nonpersistent)
        close.add(name + ':buffer registrations', registrations)
    for name, parameter in model.named_parameters():
        close.add(name + ':gradient flag', lambda p=parameter, flag=parameter.requires_grad: p.requires_grad_(flag))
        close.add(name + ':gradient object', lambda p=parameter, grad=parameter.grad: setattr(p, 'grad', grad))
        if parameter.grad is not None:
            close.add(name + ':gradient value', lambda grad=parameter.grad, saved=parameter.grad.detach().clone(): restore_value(grad, saved))
    for name, buffer in model.named_buffers():
        close.add(name + ':buffer value', lambda buffer=buffer, saved=buffer.detach().clone(): restore_value(buffer, saved))
    try:
        # Build all executable graphs before modifying the borrowed model.
        for name, module in targets:
            graphs[name.replace('.', '__')] = LoRAGraph(module, rank)
        model.requires_grad_(False).eval()
        for name, module in targets:
            graph = graphs[name.replace('.', '__')]
            def hook(module, inputs, output, graph=graph):
                if graph.multiplier == 0:
                    return output
                return output + graph.delta(inputs[0]) * graph.multiplier
            handle = module.register_forward_hook(hook)
            close.add(name + ':hook', handle.remove, first=True)
        @contextmanager
        def scaled(scale):
            previous = [g.multiplier for g in graphs.values()]
            try:
                for g in graphs.values():
                    g.multiplier = scale
                yield
            finally:
                for g, value in zip(graphs.values(), previous):
                    g.multiplier = value
        network = AdapterParameters(graphs.named_parameters(), scaled)
        return network, close
    except BaseException as primary:
        close(primary=primary)
        raise


def native(options, *, model=None):
    verify_core_closure()
    model = load_model(options) if model is None else model
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('this cohort requires CPU FP32 native base')
    identity = model_identity(model)
    network, close = attach(model, options.get('rank', 4))
    try:
        return NativeResponse(model, network, identity), close
    except BaseException as primary:
        close(primary=primary)
        raise


def prepare(options, *, model=None, rows=None):
    verify_core_closure()
    with torch.random.fork_rng():
        torch.manual_seed(options.get('seed', 7))
        return _prepare(options, model=model, rows=rows)


def _prepare(options, *, model=None, rows=None):
    model = load_model(options) if model is None else model
    if rows is None:
        if options.get('dummy', False):
            rows = fixture_rows()
        else:
            cache = torch.load(options['cache_file'], map_location='cpu', weights_only=True)
            if set(cache) != {'model_identity', 'rows'} or cache['model_identity'] != model_identity(model):
                raise ValueError('frozen cache belongs to a different Stage C base')
            rows = [ResponseRow(**row) for row in cache['rows']]
    if len(rows) < 2:
        raise ValueError('at least two frozen rows required')
    for row in rows:
        check_row(row, model, options.get('shape', [1, 4, 4, 4]))
    if not any(row.guidance > 0 for row in rows):
        raise ValueError('entirely zero-guidance training objective: guidance=0 cancels adapter influence')
    response, close = native(options, model=model)
    try:
        prepared = [response.prepare(row) for row in rows]
        conditions = unit_ratio_condition_features(prepared)
        critic = HNDLFieldCritic(prepared[0].target.shape[1], conditions.shape[1],
                                 prepared[0].target.shape[2:], width=8)
        game = ConditionalPG09(response, prepared, critic, conditions=conditions,
            total_steps=options.get('steps', 4), batch_size=options.get('batch', 2), seed=options.get('seed', 7))
        configuration = {k: v for k, v in options.items() if k in ('dummy', 'shape', 'rank')}
        spec = INFERENCE.specification(response, model_identity=response.model_identity,
            protocol=PROTOCOL, configuration=configuration)
        return Binding(game, close, inference=spec)
    except BaseException as primary:
        close(primary=primary)
        raise


def validate_inference(options, spec):
    verify_core_closure()
    if (spec.identity['protocol'] != PROTOCOL
            or spec.identity['model'].get('critic_feature_policy') != FEATURE_POLICY
            or spec.identity['model'].get('adapter_policy') != ADAPTER_POLICY):
        raise ValueError('native inference requires explicit Cascade v2 feature/activation policies; no rebinding')
    if set(options) - {'model_dir', 'base_precision'}:
        raise ValueError('native-only inference accepts local base path/precision, no training cache')
    validate(dict(spec.configuration, **options)) if spec.configuration.get('dummy') else _validate_native(options)


def _validate_native(options):
    if not Path(options.get('model_dir', '')).is_dir() or options.get('base_precision', 'float32') != 'float32':
        raise ValueError('local CPU FP32 Stage C base required')
    verified_builder()
    if version('diffusers') != '0.36.0':
        raise ValueError('Cascade requires diffusers==0.36.0')


def prepare_inference(options, spec):
    verify_core_closure()
    configuration = dict(spec.configuration, **options)
    model = load_model(configuration)
    if model_identity(model) != spec.identity['model']:
        raise ValueError('native Stage C base/source differs before adapter attachment')
    response, close = native(configuration, model=model)
    try:
        identity = native_identity(response, model_identity=response.model_identity, protocol=PROTOCOL)
        def predict(*, row, strength=1., seed=0, **request):
            if request:
                raise ValueError('unknown Stage C response arguments')
            check_row(row, response.model, spec.configuration.get('shape', [1, 4, 4, 4]))
            return response._response(row, scale=strength).detach().clone()
        return InferenceBinding(response.network, identity, close, response=predict)
    except BaseException as primary:
        close(primary=primary)
        raise


BACKEND = ResponseBackend(NAME, capabilities({}), prepare, validate, configure_capabilities=capabilities)
INFERENCE = InferenceBackend(NAME, Capabilities('adapter_gan', False), prepare_inference,
    validate_inference, configure_capabilities=capabilities)
