from experiments.benchmark_v3 import select_policy


def test_selects_sparse_fallback():
    rows = [{'label': 'POSITIVE'}, {'label': 'NEGATIVE'}]
    small = [{'label': 'POSITIVE', 'confidence': .99}, {'label': 'POSITIVE', 'confidence': .6}]
    large = [{'label': 'POSITIVE', 'confidence': .99}, {'label': 'NEGATIVE', 'confidence': .99}]
    policy, candidates, baseline = select_policy(rows, small, large, max_drop=0, max_fallback=.5)
    assert baseline == 1
    assert policy['threshold'] == .7
    assert policy['fallback_rate'] == .5


def test_infeasible_policy_returns_none():
    rows = [{'label': 'NEGATIVE'}]
    small = [{'label': 'POSITIVE', 'confidence': .99}]
    large = [{'label': 'NEGATIVE', 'confidence': .99}]
    policy, *_ = select_policy(rows, small, large, max_drop=0, max_fallback=0)
    assert policy is None
