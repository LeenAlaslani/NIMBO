"""Load a calibrated policy from a benchmark report and execute the real text agent.

No quality promise is inferred from a single validation run. The confidence score
is a model score, not a calibrated probability.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

from nimbo.core import NimboAgent, Request, Modality, Trace


@dataclass(frozen=True)
class TextPolicy:
    kind: str
    threshold: float | None
    small_model: str
    large_model: str
    source: str


def load_text_policy(path: str | Path) -> TextPolicy:
    """Read the v4 selection made on validation only, never the test metrics."""
    p = Path(path)
    report = json.loads(p.read_text(encoding='utf-8'))
    selected = report['selected_policy']
    models = report['models']
    kind = selected['type']
    if kind not in ('adaptive', 'small_only'):
        raise ValueError(f'Unsupported text policy: {kind!r}')
    threshold = selected.get('threshold')
    if kind == 'adaptive' and (not isinstance(threshold, (float, int)) or isinstance(threshold, bool) or not 0.5 <= threshold <= 1.0):
        raise ValueError('Adaptive policy threshold must be in [0.5, 1.0]')
    if kind == 'small_only' and threshold is not None:
        raise ValueError('Small-only policy must not have a threshold')
    if not models.get('small') or not models.get('large'):
        raise ValueError('Report must identify both checkpoint IDs')
    return TextPolicy(kind, float(threshold) if threshold is not None else None,
                      models['small'], models['large'], str(p))


def run_text_policy(text: str, policy: TextPolicy, small, large) -> Trace:
    """Run through the same NimboAgent as the app; trace records each step."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Text must be a nonempty string')
    if policy.kind == 'small_only':
        def unexpected_large(_):
            raise RuntimeError('Small-only policy must never execute large model')
        agent = NimboAgent(text_small=small, text_large=unexpected_large,
                           image_fast=lambda _: None, image_quality=lambda _: None,
                           confidence_threshold=0.0)
    elif policy.kind == 'adaptive':
        agent = NimboAgent(text_small=small, text_large=large,
                           image_fast=lambda _: None, image_quality=lambda _: None,
                           confidence_threshold=policy.threshold)
    else:
        raise ValueError(f'Unsupported kind: {policy.kind}')
    trace = agent.run(Request(modality=Modality.TEXT, payload=text))
    trace.events.insert(0, {'state': 'policy_loaded', 'policy': policy.kind,
                             'threshold': policy.threshold, 'source': policy.source,
                             'small_model': policy.small_model, 'large_model': policy.large_model,
                             'note': 'confidence is uncalibrated classifier score'})
    if policy.kind == 'small_only':
        trace.decision['route'] = 'small-only'
        trace.decision['reason'] = 'Calibration selected small-only; no large-model call'
    return trace


def load_and_run_text(text: str, report_path: str = 'results/rotten_tomatoes_v4.json', device: int = -1) -> Trace:
    """Convenience entrypoint; models are loaded on demand (cache them in a server)."""
    from nimbo.text_backend import load_classifier
    policy = load_text_policy(report_path)
    small = load_classifier(policy.small_model, device=device, sst2_labels=True)
    large = (load_classifier(policy.large_model, device=device, sst2_labels=True)
             if policy.kind == 'adaptive' else lambda _: None)
    return run_text_policy(text, policy, small, large)
