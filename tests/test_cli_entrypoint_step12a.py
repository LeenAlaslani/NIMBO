"""Regression: CLI launch must work both as module and as direct script.

Only tests --help; does not load/download pretrained models or modify logs.
"""
import subprocess
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('name', ['resource_objective_v2', 'staged_candidate_smoke_v1'])
@pytest.mark.parametrize('launch', ['direct', 'module'])
def test_cli_entrypoint_imports_project_packages(name, launch):
    args = ([sys.executable, str(ROOT/'tools'/f'{name}.py'), '--help']
            if launch == 'direct' else
            [sys.executable, '-m', f'tools.{name}', '--help'])
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=40)
    assert result.returncode == 0, f'{launch} {name}: {result.stderr}'
    assert 'usage:' in result.stdout.lower()
