from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference
from particle_sliders_cascade import backend as b
from particle_sliders_cascade import execution as e


@pytest.mark.parametrize('options', [dict(device='cuda:0'), dict(runtime_profile='unknown'),
    dict(runtime_profile=e.CUDA_PROFILE, device='cpu'), dict(base_precision='bfloat16')])
def test_execution_profile_refuses_ambiguous_placement_before_allocation(options, monkeypatch):
    monkeypatch.setattr(b, 'load_model', lambda *args: pytest.fail('allocated before profile admission'))
    with pytest.raises((ValueError, UnsupportedOperation)):
        open_response(SessionConfig(b.NAME, 'bad-placement', dummy=True, **options))


def test_cpu_constructor_does_not_snapshot_or_initialize_visible_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda *args: pytest.fail('CPU path touched CUDA RNG'))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: pytest.fail('CPU path inspected CUDA devices'))
    model = b.tiny_model()
    with open_response(SessionConfig(b.NAME, 'cpu-only', dummy=True), model=model) as session:
        assert session.game.backend.model_identity['execution_policy'] == e.execution_policy({})
        session.step()


def test_wrong_execution_identity_refused_before_inference_allocation(monkeypatch):
    with open_response(SessionConfig(b.NAME, 'source-policy', dummy=True)) as session:
        artifact = session.export_inference()
    artifact['specification']['identity']['model']['execution_policy']['profile'] = e.CUDA_PROFILE
    artifact['sha256'] = b._hash({key: value for key, value in artifact.items() if key != 'sha256'})
    monkeypatch.setattr(b, 'load_model', lambda *args: pytest.fail('allocated before policy rejection'))
    with pytest.raises(ValueError, match='execution policy differs'):
        open_inference(artifact, operation='response')


def test_caller_flags_are_checked_without_enabling_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('profile initialized CUDA'))
    # Deliberately invalid regardless of the surrounding process settings.
    monkeypatch.setattr(torch, 'are_deterministic_algorithms_enabled', lambda: False)
    with pytest.raises(ValueError, match='caller-set strict'):
        e.execution_policy(dict(runtime_profile=e.CUDA_PROFILE, device='cuda:0'))


def test_native_checkpointing_captures_signed_scales_and_restores_borrowed_model():
    torch.set_num_threads(1)
    model = b.tiny_model()
    rows = b.fixture_rows()
    rows[1] = replace(rows[1], scale=-1.)
    original = model._gradient_checkpointing_func
    assert not model.is_gradient_checkpointing
    outputs, gradients = [], []
    for checkpointing in (False, True):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            response, close = b.native(dict(dummy=True, gradient_checkpointing=checkpointing), model=model)
        try:
            with torch.no_grad():
                for p in response.network.parameters():
                    p.add_(.01)
            # Construct both forwards before backward: recomputation must retain
            # each forward's scale even after the scale context has unwound.
            values = [response._response(row, scale=strength) for row, strength in zip(rows, (.5, -1.))]
            loss = sum(value.square().mean() for value in values)
            leaves = list(response.network.parameters())
            grad = torch.autograd.grad(loss, leaves)
            assert all(torch.isfinite(value).all() for value in grad)
            outputs.append([value.detach() for value in values])
            gradients.append([value.detach() for value in grad])
        finally:
            close()
        assert not model.is_gradient_checkpointing
        assert model._gradient_checkpointing_func is original
    for actual, expected in zip(outputs[0] + gradients[0], outputs[1] + gradients[1]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_v2_packets_require_original_source_instead_of_rebinding(monkeypatch):
    with open_response(SessionConfig(b.NAME, 'fresh-v3', dummy=True)) as session:
        artifact = deepcopy(session.export_inference())
    artifact['specification']['identity']['protocol'] = 'cascade-stage-c-spatial-response-hndl-v2'
    artifact['sha256'] = b._hash({key: value for key, value in artifact.items() if key != 'sha256'})
    monkeypatch.setattr(b, 'load_model', lambda *args: pytest.fail('allocated old source cohort'))
    with pytest.raises(ValueError, match='requires explicit Cascade v3'):
        open_inference(artifact, operation='response')
