from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from particle_sliders.lifecycle import SessionConfig, CheckpointError, UnsupportedOperation
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference
from particle_sliders.fibo_hndl import HNDLFieldCritic, verified_builder
from particle_sliders.field_critic import FieldCritic
from particle_sliders_cascade import backend as b

torch.set_num_threads(1)


def config(**options):
    return SessionConfig(b.NAME, 'cascade-new-cpu-test', dummy=True, steps=3, ema_decay=.9, **options)


def snapshot(model):
    return ([m.training for m in model.modules()], [(p.requires_grad, p.grad) for p in model.parameters()],
            [len(m._forward_hooks) for m in model.modules()])


def test_installed_identity():
    assert 'site-packages' in str(Path(b.__file__))
    import particle_sliders.pg09
    assert 'site-packages' in particle_sliders.pg09.__file__
    assert verified_builder()[1]['hndl_version'] == '0.10.0'
    assert b.PG_VERSION == '0.9.4'


def test_hndl_lora_outputs_vjps():
    graph = b.LoRAGraph(nn.Linear(5, 7), 3)
    nn.init.normal_(graph.graph['up'].weight)
    x = torch.randn(2, 4, 5, requires_grad=True)
    actual = graph.delta(x)
    reference = torch.nn.functional.linear(torch.nn.functional.linear(x, graph.graph['down'].weight),
                                            graph.graph['up'].weight)
    torch.testing.assert_close(actual, reference)
    cotangent = torch.randn_like(actual)
    leaves = [x, *graph.parameters()]
    for got, wanted in zip(torch.autograd.grad(actual, leaves, cotangent, retain_graph=True),
                           torch.autograd.grad(reference, leaves, cotangent)):
        torch.testing.assert_close(got, wanted)


def test_hndl_critic_second_derivatives():
    actual = HNDLFieldCritic(4, 6, (4, 4), width=8)
    reference = FieldCritic(4, 6, width=8)
    pairs = [('field0', reference.field[0]), ('field2', reference.field[2]),
             ('condition', reference.condition), ('score0', reference.score[0]), ('score2', reference.score[2])]
    for name, layer in pairs:
        layer.load_state_dict(actual.graph[name].state_dict())
    x = torch.randn(2, 4, 4, 4, requires_grad=True)
    context = torch.randn(2, 6, requires_grad=True)
    left, right = actual(x, context), reference(x, context)
    torch.testing.assert_close(left, right)
    ga = torch.autograd.grad(left.sum(), (x, context), create_graph=True)
    gr = torch.autograd.grad(right.sum(), (x, context), create_graph=True)
    for a, r in zip(ga, gr):
        torch.testing.assert_close(a, r)
    actual_parameters = [p for name, _ in pairs for p in actual.graph[name].parameters()]
    reference_parameters = [p for _, layer in pairs for p in layer.parameters()]
    ha = torch.autograd.grad(sum(v.square().sum() for v in ga), [x, context, *actual_parameters], allow_unused=True)
    hr = torch.autograd.grad(sum(v.square().sum() for v in gr), [x, context, *reference_parameters], allow_unused=True)
    for a, r in zip(ha, hr):
        if a is None or r is None:
            assert a is r
        else:
            torch.testing.assert_close(a, r, atol=1e-7, rtol=1e-5)


def test_native_response_and_strength_restore():
    model, rows = b.tiny_model(), b.fixture_rows()
    model.train()
    next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
    before = snapshot(model)
    with open_response(config(), model=model, rows=rows) as session:
        with torch.no_grad():
            native = model(sample=rows[0].latent, timestep_ratio=rows[0].timestep.reshape(1),
                clip_text=rows[0].neutral['embeds'], clip_text_pooled=rows[0].neutral['pooled'],
                clip_img=rows[0].neutral['image_embeds']).sample
        torch.testing.assert_close(session.evaluate(strength=0), native, rtol=0, atol=0)
        original = [p.clone() for p in session.game.network.parameters()]
        rng = torch.get_rng_state().clone()
        stats = session.step()
        assert stats['step'] == 1 and stats['d_pen'] >= 0
        assert any(not torch.equal(a, p) for a, p in zip(original, session.game.network.parameters()))
        zero = session.evaluate(strength=0)
        positive, negative = session.evaluate(strength=1), session.evaluate(strength=-1)
        assert not torch.equal(positive, zero) and not torch.equal(negative, positive)
        assert torch.equal(torch.get_rng_state(), rng)
    assert snapshot(model) == before


def test_exact_resume_ema_native_only(tmp_path, monkeypatch):
    c = config()
    with open_response(c) as full:
        full.train()
        wanted = full.checkpoint()
    with open_response(c) as first:
        first.train(stop_after=1)
        first.checkpoint(tmp_path / 'resume.pt')
    with open_response(c, resume=tmp_path / 'resume.pt') as resumed:
        resumed.train()
        assert b._hash(resumed.checkpoint()) == b._hash(wanted)
        exported = resumed.export_inference(selection='ema')
        with torch.no_grad():
            saved = deepcopy(resumed.game.network.state_dict())
            resumed.game.network.load_state_dict(resumed.ema.values)
            expected = resumed.evaluate(strength=.5)
            resumed.game.network.load_state_dict(saved)
    def forbidden(*args, **kwargs):
        raise AssertionError('training object constructed during native inference')
    monkeypatch.setattr(b, 'ConditionalPG09', forbidden)
    monkeypatch.setattr(b, 'HNDLFieldCritic', forbidden)
    monkeypatch.setattr(b, 'fixture_rows', forbidden)
    with open_inference(exported, operation='response') as inference:
        assert not hasattr(inference, 'game')
        # Independent caller supplied row, no training cache in artifact.
        row = FIXTURE()[0]
        torch.testing.assert_close(inference.response(row=row, strength=.5), expected, rtol=0, atol=0)
        with pytest.raises(UnsupportedOperation):
            inference.train()


FIXTURE = b.fixture_rows


def test_preflight_and_failure_cleanup(monkeypatch):
    model, rows = b.tiny_model(), b.fixture_rows()
    before = snapshot(model)
    with pytest.raises(UnsupportedOperation):
        open_response(config(), operation='render', model=model, rows=rows)
    with pytest.raises(ValueError):
        open_response(config(), model=model, rows=[replace(rows[0], timestep=torch.tensor(100.)), rows[1]])
    assert snapshot(model) == before
    def fail(*args, **kwargs):
        raise RuntimeError('critic allocation failed')
    monkeypatch.setattr(b, 'HNDLFieldCritic', fail)
    with pytest.raises(RuntimeError, match='critic allocation'):
        open_response(config(), model=model, rows=rows)
    assert snapshot(model) == before


def test_rejection_rollback_and_failed_update(monkeypatch):
    model = b.tiny_model()
    before = snapshot(model)
    session = open_response(config(), model=model)
    session.step()
    state = session.checkpoint()
    corrupted = deepcopy(state)
    corrupted['payload']['game']['completed_steps'] = 99
    corrupted['sha256'] = b._hash({k: v for k, v in corrupted.items() if k != 'sha256'})
    with pytest.raises(CheckpointError):
        session.restore(corrupted)
    assert b._hash(session.checkpoint()) == b._hash(state)
    def fail(*args, **kwargs):
        raise RuntimeError('native forward failed')
    monkeypatch.setattr(model, 'forward', fail)
    with pytest.raises(RuntimeError, match='native forward'):
        session.step()
    assert session.closed and snapshot(model) == before
    session.close()


def test_cross_cohort_rejects_before_model_allocation(tmp_path, monkeypatch):
    with open_response(config()) as session:
        session.checkpoint(tmp_path / 'checkpoint.pt')
    monkeypatch.setattr(b, 'load_model', lambda *args: pytest.fail('allocated before rejection'))
    with pytest.raises(CheckpointError):
        open_response(SessionConfig(b.NAME, 'different', dummy=True, steps=3, ema_decay=.9),
                      resume=tmp_path / 'checkpoint.pt')


@pytest.mark.parametrize('stage', ['graph', 'teacher', 'game', 'callback'])
def test_every_partial_failure_unwinds(stage, monkeypatch):
    model = b.tiny_model()
    model.train()
    before = snapshot(model)
    calls = 0
    original = b.LoRAGraph
    def graph(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected graph')
        return original(*args, **kwargs)
    def fail(*args, **kwargs):
        raise RuntimeError('injected ' + stage)
    if stage == 'graph':
        monkeypatch.setattr(b, 'LoRAGraph', graph)
    elif stage == 'teacher':
        monkeypatch.setattr(b.NativeResponse, 'prepare', fail)
    elif stage == 'game':
        monkeypatch.setattr(b, 'ConditionalPG09', fail)
    with pytest.raises(RuntimeError, match='injected'):
        with open_response(config(), model=model) as session:
            session.train(on_step=fail)
    assert snapshot(model) == before


def test_local_staging_and_cache_identity(tmp_path, monkeypatch):
    from dataclasses import asdict
    model_dir = tmp_path / 'prior'
    b.tiny_model().save_pretrained(model_dir, safe_serialization=True)
    options = dict(model_dir=str(model_dir), shape=[1, 4, 4, 4], steps=2)
    model = b.load_model(options)
    cache = tmp_path / 'frozen.pt'
    torch.save(dict(model_identity=b.model_identity(model), rows=[asdict(r) for r in FIXTURE()]), cache)
    options.update(cache_file=str(cache), cache_sha256=b.hashlib.sha256(cache.read_bytes()).hexdigest())
    with open_response(SessionConfig(b.NAME, 'staged-tiny', **options)) as session:
        session.step()
        artifact = session.export_inference()
    assert 'cache_file' not in artifact['specification']['configuration']
    with open_inference(artifact, options={'model_dir': str(model_dir)}, operation='response') as native:
        assert native.response(row=FIXTURE()[0]).shape == (1, 4, 4, 4)
    changed = b.load_model(options)
    with torch.no_grad():
        next(changed.parameters()).add_(.1)
    before = snapshot(changed)
    monkeypatch.setattr(b, 'load_model', lambda *args: changed)
    with pytest.raises(ValueError, match='before adapter attachment'):
        open_inference(artifact, options={'model_dir': str(model_dir)}, operation='response')
    assert snapshot(changed) == before
    with pytest.raises(ValueError, match='different Stage C base'):
        open_response(SessionConfig(b.NAME, 'wrong-cache-base', **options))
    assert snapshot(changed) == before


@pytest.mark.parametrize('failure', [None, 'open', 'response'])
def test_public_inference_rng_success_and_failure(failure, monkeypatch):
    with open_response(config()) as session:
        artifact = session.export_inference()
    original_loader = b.load_model
    def load(options):
        torch.rand(5)
        if failure == 'open':
            raise RuntimeError('injected RNG open failure')
        model = original_loader(options)
        original_forward = model.forward
        def forward(*args, **kwargs):
            torch.rand(7)
            if failure == 'response':
                raise RuntimeError('injected RNG response failure')
            return original_forward(*args, **kwargs)
        model.forward = forward
        return model
    monkeypatch.setattr(b, 'load_model', load)
    before = torch.get_rng_state().clone()
    if failure == 'open':
        with pytest.raises(RuntimeError, match='RNG open failure'):
            open_inference(artifact, operation='response')
        assert torch.equal(torch.get_rng_state(), before)
        return
    with open_inference(artifact, operation='response') as inference:
        assert torch.equal(torch.get_rng_state(), before)
        if failure == 'response':
            with pytest.raises(RuntimeError, match='RNG response failure'):
                inference.response(row=FIXTURE()[0])
        else:
            inference.response(row=FIXTURE()[0])
        assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(torch.get_rng_state(), before)


def test_unit_ratio_features_preserve_native_calls_cfg_and_zero():
    model = b.tiny_model()
    rows = [replace(FIXTURE()[0], timestep=torch.tensor(r), guidance=2.) for r in (.25, .75)]
    ratios = []
    handle = model.register_forward_pre_hook(
        lambda module, args, kwargs: ratios.append(float(kwargs['timestep_ratio'])), with_kwargs=True)
    try:
        with open_response(config(), model=model, rows=rows) as session:
            assert session.game.backend.metadata()['critic_feature_policy'] == b.FEATURE_POLICY
            torch.testing.assert_close(session.game.conditions[:, 0], torch.tensor([.25, .75]), rtol=0, atol=0)
            assert float(session.game.conditions[1, 0] - session.game.conditions[0, 0]) == .5
            for item in session.game.rows:
                row = item.row
                assert float(row.timestep) in (.25, .75)
                def raw(condition):
                    with torch.no_grad(), session.game.network.scaled(0.):
                        return model(sample=row.latent, timestep_ratio=row.timestep.reshape(1),
                            clip_text=condition['embeds'], clip_text_pooled=condition['pooled'],
                            clip_img=condition['image_embeds']).sample
                negative = raw(row.negative)
                expected_target = negative + row.guidance * (raw(row.positive) - negative)
                expected_neutral = negative + row.guidance * (raw(row.neutral) - negative)
                torch.testing.assert_close(item.target, expected_target, rtol=0, atol=0)
                torch.testing.assert_close(item.neutral, expected_neutral, rtol=0, atol=0)
            session.step()
            for i, item in enumerate(session.game.rows):
                torch.testing.assert_close(session.evaluate(index=i, strength=0), item.neutral, rtol=0, atol=0)
            assert session.binding.inference.identity['protocol'] == 'cascade-stage-c-spatial-response-hndl-v2'
        assert set(ratios) == {.25, .75}
        assert [float(r.timestep) for r in rows] == [.25, .75]
    finally:
        handle.remove()


@pytest.mark.parametrize('mutation', ['python', 'data', 'missing', 'nested-python', 'nested-data', 'compatibility-python'])
def test_recursive_core_closure_before_native_mutation(tmp_path, monkeypatch, mutation):
    import shutil
    with open_response(config()) as session:
        artifact = session.export_inference()
    copy = tmp_path / 'core'
    shutil.copytree(b.core_root(), copy / 'particle_sliders', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(b.core_root().parent / 'concept_slider_core', copy / 'concept_slider_core',
                    ignore=shutil.ignore_patterns('__pycache__'))
    if mutation == 'python':
        (copy / 'particle_sliders/lifecycle.py').write_text((copy / 'particle_sliders/lifecycle.py').read_text() + '\n# altered\n')
    elif mutation == 'data':
        (copy / 'particle_sliders/fibo_critic.hndl').write_text('altered graph\n')
    elif mutation == 'missing':
        (copy / 'particle_sliders/pg094_release.json').unlink()
    elif mutation == 'compatibility-python':
        (copy / 'concept_slider_core/__init__.py').write_text('# altered compatibility namespace\n')
    else:
        (copy / 'particle_sliders/nested').mkdir()
        (copy / 'particle_sliders/nested' / ('extra.py' if mutation == 'nested-python' else 'extra.json')).write_text('{}')
    monkeypatch.setattr(b, 'core_root', lambda: copy / 'particle_sliders')
    monkeypatch.setattr(b, 'load_model', lambda *args: pytest.fail('native allocation before closure rejection'))
    with pytest.raises(ValueError, match='recursive shared-core source/data closure'):
        open_response(config())
    with pytest.raises(ValueError, match='recursive shared-core source/data closure'):
        open_inference(artifact, operation='response')


def test_cleanup_failure_before_hook_removal_is_best_effort_and_retryable(monkeypatch):
    from torch.utils.hooks import RemovableHandle
    model = b.tiny_model()
    model.train()
    list(model.modules())[-1].eval()
    parameters = list(model.parameters())
    parameters[0].requires_grad_(False)
    gradient = torch.ones_like(parameters[1])
    parameters[1].grad = gradient
    buffer = torch.tensor([3.])
    model.register_buffer('audit_buffer', buffer)
    model.gradient_checkpointing = True
    checkpoint_function = model._gradient_checkpointing_func
    before = snapshot(model)
    remove = RemovableHandle.remove
    attempts = []
    def flaky(handle):
        attempts.append(handle.id)
        if len(attempts) == 1:
            raise RuntimeError('hook failure BEFORE removal')
        return remove(handle)
    monkeypatch.setattr(RemovableHandle, 'remove', flaky)
    _, close = b.attach(model, 4)
    parameters[1].grad = torch.zeros_like(gradient)
    gradient.fill_(8)
    buffer.add_(9)
    model.register_buffer('extra_buffer', torch.ones(1))
    model.gradient_checkpointing = False
    model._gradient_checkpointing_func = object()
    with pytest.raises(BaseExceptionGroup, match='cleanup incomplete'):
        close()
    # A hook failure must not prevent any modes/flags/gradient/buffer restoration.
    after = snapshot(model)
    assert after[:2] == before[:2]
    assert sum(after[2]) == sum(before[2]) + 1
    assert parameters[1].grad is gradient
    torch.testing.assert_close(gradient, torch.ones_like(gradient))
    assert model.audit_buffer is buffer and float(buffer) == 3.
    assert not hasattr(model, 'extra_buffer') and model._gradient_checkpointing_func is checkpoint_function
    assert model.gradient_checkpointing is True and close.pending and len(close.errors) == 1
    close()
    assert not close.pending and snapshot(model) == before
    count = len(attempts)
    close()
    assert len(attempts) == count


def test_cleanup_preserves_original_failure_and_retains_retry(monkeypatch):
    from torch.utils.hooks import RemovableHandle
    model = b.tiny_model()
    before = snapshot(model)
    original = RuntimeError('original teacher exception')
    remove = RemovableHandle.remove
    attempts, callbacks = [], []
    def flaky(handle):
        attempts.append(handle.id)
        if len(attempts) == 1:
            raise RuntimeError('injected removal failure')
        return remove(handle)
    attach = b.attach
    def remember(*args, **kwargs):
        network, close = attach(*args, **kwargs)
        callbacks.append(close)
        return network, close
    def fail(*args, **kwargs):
        raise original
    monkeypatch.setattr(RemovableHandle, 'remove', flaky)
    monkeypatch.setattr(b, 'attach', remember)
    monkeypatch.setattr(b.NativeResponse, 'prepare', fail)
    with pytest.raises(RuntimeError) as caught:
        open_response(config(), model=model)
    assert caught.value is original
    assert any('cleanup incomplete' in note for note in original.__notes__)
    assert callbacks[0].pending
    assert original.cascade_cleanup is callbacks[0]
    original.cascade_cleanup()
    assert snapshot(model) == before and not callbacks[0].pending


def test_cleanup_in_unrelated_except_reports_its_own_failure_and_retries(monkeypatch):
    from torch.utils.hooks import RemovableHandle
    model = b.tiny_model()
    before = snapshot(model)
    remove = RemovableHandle.remove
    attempts = []
    def flaky(handle):
        attempts.append(handle.id)
        if len(attempts) == 1:
            raise RuntimeError('cleanup failed before hook removal')
        return remove(handle)
    monkeypatch.setattr(RemovableHandle, 'remove', flaky)
    _, close = b.attach(model, 4)
    unrelated = ValueError('unrelated caller exception')
    try:
        raise unrelated
    except ValueError:
        with pytest.raises(BaseExceptionGroup, match='cleanup incomplete'):
            close()
        assert not hasattr(unrelated, 'cascade_cleanup')
        assert not hasattr(unrelated, '__notes__')
    assert close.pending and len(close.errors) == 1
    assert snapshot(model)[:2] == before[:2]
    close()
    assert snapshot(model) == before and not close.pending


def test_existing_raw_host_on_and_public_signed_strength_policy():
    model, row = b.tiny_model(), FIXTURE()[0]
    with open_response(config(), model=model) as session:
        session.step()
        with torch.no_grad():
            bare = model(sample=row.latent, timestep_ratio=row.timestep.reshape(1),
                clip_text=row.neutral['embeds'], clip_text_pooled=row.neutral['pooled'],
                clip_img=row.neutral['image_embeds']).sample
        torch.testing.assert_close(bare, session.evaluate(strength=1), rtol=0, atol=0)
        assert not torch.equal(bare, session.evaluate(strength=0))
        torch.testing.assert_close(session.evaluate(), session.evaluate(strength=1), rtol=0, atol=0)
        for strength in (-2., -.5, 0., 1., 2.):
            assert torch.isfinite(session.evaluate(strength=strength)).all()
        for strength in (float('nan'), float('inf')):
            with pytest.raises(UnsupportedOperation):
                session.evaluate(strength=strength)
        artifact = session.export_inference()
        policy = artifact['specification']['identity']['model']['adapter_policy']
        assert policy == b.ADAPTER_POLICY and policy['raw_host_default'] == 1
    with open_inference(artifact, operation='response') as inference:
        torch.testing.assert_close(inference.response(row=row), inference.response(row=row, strength=1), rtol=0, atol=0)
        for strength in (-2., 2.):
            assert torch.isfinite(inference.response(row=row, strength=strength)).all()


def test_guidance_zero_cancels_gradients_and_training_rejects_before_hooks():
    model, row = b.tiny_model(), replace(FIXTURE()[0], guidance=0.)
    before = snapshot(model)
    response, close = b.native({'dummy': True}, model=model)
    try:
        value = response._response(row)
        gradients = torch.autograd.grad(value.sum(), list(response.network.parameters()), allow_unused=True)
        assert all(g is not None and torch.count_nonzero(g) == 0 for g in gradients)
    finally:
        close()
    assert snapshot(model) == before
    with pytest.raises(ValueError, match='guidance=0 cancels adapter influence'):
        open_response(config(), model=model, rows=[row, replace(row, timestep=torch.tensor(.75))])
    assert snapshot(model) == before


def test_old_native_protocol_rejected_before_allocation(monkeypatch):
    with open_response(config()) as session:
        artifact = session.export_inference()
    artifact['specification']['identity']['protocol'] = 'cascade-stage-c-spatial-response-hndl-v1'
    artifact['sha256'] = b._hash({key: value for key, value in artifact.items() if key != 'sha256'})
    monkeypatch.setattr(b, 'load_model', lambda *args: pytest.fail('allocated before old protocol rejection'))
    with pytest.raises(ValueError, match='requires explicit Cascade v2'):
        open_inference(artifact, operation='response')


def test_mixed_zero_guidance_retains_frozen_negative_control():
    rows = FIXTURE()
    rows[0] = replace(rows[0], guidance=0.)
    with open_response(config(), rows=rows) as session:
        item = session.game.rows[0]
        expected = session.game.backend.velocity(item.row.negative, item.row.latent, item.row.timestep, scale=0)
        torch.testing.assert_close(item.target, expected, rtol=0, atol=0)
        torch.testing.assert_close(item.neutral, expected, rtol=0, atol=0)
        before = [p.clone() for p in session.game.network.parameters()]
        session.step()
        assert any(not torch.equal(a, p) for a, p in zip(before, session.game.network.parameters()))
        torch.testing.assert_close(session.evaluate(index=0, strength=2), expected, rtol=0, atol=0)


def test_shared_origin_refused_before_native_allocation(monkeypatch):
    installed = b.distribution
    class WrongOrigin:
        def read_text(self, name):
            assert name == 'direct_url.json'
            return '{"url":"https://example.invalid/other.git","vcs_info":{"commit_id":"changed"}}'
    monkeypatch.setattr(b, 'distribution', lambda name: WrongOrigin() if name == 'particle-sliders-core' else installed(name))
    monkeypatch.setattr(b, 'load_model', lambda *args: pytest.fail('native allocation before source admission'))
    with pytest.raises(ValueError, match='exact pinned shared-core Git origin'):
        open_response(config())


def test_adapter_only_particle_branch_preserved():
    with open_response(config()) as session:
        assert session.game.prior is None
        assert session.game.spread is None
        assert session.game.recipe.total_steps == 3
        session.step()
        assert session.game.completed_steps == 1
