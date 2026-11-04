"""Installed shared executor conformance for the explicitly synthetic Stage C arm."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def test_unified_fixture_periodic_exact_resume_and_exports(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    configuration = json.loads((repository / 'tiny-config.json').read_text())
    profile = json.loads((repository / 'tiny-profile.json').read_text())
    (tmp_path / 'config.json').write_text(json.dumps(configuration))
    (tmp_path / 'profile.json').write_text(json.dumps(profile))
    command = str(Path(sys.executable).with_name('particle-slider'))
    environment = dict(os.environ, PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def cli(*arguments, success=True):
        result = subprocess.run([command, *arguments], cwd=tmp_path, env=environment,
                                text=True, capture_output=True, timeout=90)
        assert (result.returncode == 0) is success, (arguments, result.stdout, result.stderr)
        return result
    common = ['--config', 'config.json', '--profile', 'profile.json', '--no-reporting',
              '--runtime-python', sys.executable]
    plan = json.loads(cli('plan', *common, '--out', 'planned').stdout)
    assert plan['resolved']['profile']['status'] == 'fixture'
    assert not (tmp_path / 'planned').exists()
    checked = json.loads(cli('train', *common, '--check', '--out', 'checked').stdout)
    assert checked['status'] == 'checked' and checked['qualification'] == 'fixture'
    assert not (tmp_path / 'checked').exists()
    refused = dict(configuration, options=dict(configuration['options'], dummy=False))
    (tmp_path / 'not-fixture.json').write_text(json.dumps(refused))
    error = cli('train', '--config', 'not-fixture.json', '--profile', 'profile.json',
                '--no-reporting', '--check', '--out', 'refused', success=False)
    assert 'dummy=true' in error.stderr
    assert not (tmp_path / 'refused').exists()
    cli('train', *common, '--out', 'partial', '--stop-after', '2', '--checkpoint-every', '1',
        '--selection', 'ema', '--export-format', 'both')
    partial = json.loads((tmp_path / 'partial/result.json').read_text())
    assert partial['status'] == 'checkpointed' and partial['completed_steps'] == 2
    for step in (1, 2):
        assert (tmp_path / f'partial/artifacts/checkpoints/step-{step:06d}/checkpoint.json').is_file()
    cli('train', *common, '--out', 'resumed', '--resume', partial['checkpoint'],
        '--checkpoint-every', '2', '--selection', 'ema', '--export-format', 'both')
    resumed = json.loads((tmp_path / 'resumed/result.json').read_text())
    assert resumed['status'] == 'completed' and resumed['completed_steps'] == resumed['target_steps'] == 4
    assert resumed['qualification'] == 'fixture'
    assert partial['semantic_sha256'] == resumed['semantic_sha256']
    final = torch.load(Path(resumed['checkpoint']) / 'training.pt', weights_only=False)
    cli('train', *common, '--out', 'uninterrupted', '--checkpoint-every', '2',
        '--selection', 'ema', '--export-format', 'both')
    uninterrupted = json.loads((tmp_path / 'uninterrupted/result.json').read_text())
    expected = torch.load(Path(uninterrupted['checkpoint']) / 'training.pt', weights_only=False)
    assert expected['sha256'] == final['sha256']
    manifest = json.loads((tmp_path / 'resumed/artifacts/manifest.json').read_text())
    assert set(manifest['files']) == {'adapter.pt', 'adapter-v2.pt'}
    assert not torch.cuda.is_initialized()
