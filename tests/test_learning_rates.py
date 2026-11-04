"""LR overrides reach real optimizers and remain exact resume identity."""
from copy import deepcopy
import json
import pytest
from particle_sliders.lifecycle import SessionConfig,CheckpointError
from particle_sliders.response import open_response
from particle_sliders_cascade import backend as product


def config(**rates):
    return SessionConfig(product.NAME,'cascade-explicit-lr',dummy=True,steps=3,**rates)


def test_optimizer_rates_and_exact_resume():
    rates=dict(lr=.001,d_lr_mult=2.)
    # This roundtrip is exactly the JSON configuration consumed by common CLI.
    configured=SessionConfig.from_dict(json.loads(json.dumps(config(**rates).as_dict())))
    with open_response(configured) as current:
        assert current.game.prior is None
        assert current.game.base_lrs[0] and set(current.game.base_lrs[0])=={.001}
        assert current.game.base_lrs[1] and set(current.game.base_lrs[1])=={.002}
        current.train(stop_after=1)
        packet=current.checkpoint()
        expected=current.train()
        final=current.checkpoint()
    with open_response(configured) as restored:
        restored.restore(packet)
        assert restored.train()==expected
        from particle_sliders.pg09 import _hash
        assert _hash(restored.checkpoint())==_hash(final)
    with open_response(config(lr=.002,d_lr_mult=2.)) as changed:
        with pytest.raises(CheckpointError):
            changed.restore(packet)


@pytest.mark.parametrize('rates',[dict(lr=0),dict(lr=True),dict(lr=float('nan')),
    dict(d_lr_mult=-1),dict(d_lr_mult=float('inf')),dict(prior_lr_mult=1.)])
def test_invalid_rates_refuse_before_allocation(rates,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('native allocation must not be reached')
    monkeypatch.setattr(product,'verify_core_closure',forbidden)
    monkeypatch.setattr(product,'load_model',forbidden)
    with pytest.raises(ValueError,match='positive|particle bank'):
        product.prepare(dict(dummy=True,**rates))


def test_omitted_rates_preserve_upstream_defaults():
    from particlegan import get_recipe
    recipe=get_recipe('bispf_optimized')
    with open_response(config()) as session:
        assert session.game.recipe.lr==recipe.lr
        assert session.game.recipe.d_lr_mult==recipe.d_lr_mult


def test_common_cli_uses_identical_configured_rates(tmp_path):
    from pathlib import Path
    import subprocess,sys,torch
    source=Path(__file__).resolve().parents[1]
    configuration=tmp_path/'config.json'
    configuration.write_text(json.dumps(config(lr=.001,d_lr_mult=2.).as_dict()))
    output=tmp_path/'cli-run'
    result=subprocess.run([str(Path(sys.executable).parent/'particle-slider'),'train',
        '--profile',str(source/'tiny-profile.json'),'--config',str(configuration),
        '--out',str(output),'--no-reporting','--selection','raw','--checkpoint-every','1'],
        cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    packet=torch.load(output/'artifacts/checkpoints/step-000003/training.pt',weights_only=True)
    recipe=packet['payload']['game']['config']['recipe']
    assert recipe['lr']==.001 and recipe['d_lr_mult']==2.
