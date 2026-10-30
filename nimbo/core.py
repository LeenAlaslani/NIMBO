"""A lightweight optimization controller. No fabricated performance measurements."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from time import perf_counter
from typing import Callable, Any

class Modality(str, Enum):
    TEXT = 'text'
    IMAGE = 'image'

@dataclass
class Request:
    modality: Modality
    payload: str
    quality_mode: str = 'balanced'  # preview, balanced, high

@dataclass
class Decision:
    route: str
    reason: str

@dataclass
class Trace:
    request: dict
    decision: dict
    events: list[dict]
    result: Any
    total_latency_ms: float
    estimated_cost_usd: float | None = None

class NimboAgent:
    """Real routing + execution. Model callbacks are supplied by integrations."""
    def __init__(self, text_small: Callable, text_large: Callable,
                 image_fast: Callable, image_quality: Callable,
                 confidence_threshold: float = 0.80):
        self.text_small = text_small
        self.text_large = text_large
        self.image_fast = image_fast
        self.image_quality = image_quality
        self.confidence_threshold = confidence_threshold

    def run(self, req: Request) -> Trace:
        start = perf_counter()
        events = []
        def log(state: str, **details):
            events.append({'state': state, 'elapsed_ms': round((perf_counter()-start)*1000,2), **details})
        log('inspect', modality=req.modality.value, quality_mode=req.quality_mode)
        if req.modality == Modality.TEXT:
            decision = Decision('small-first', 'Try smaller classifier, escalate when confidence is low')
            log('select', route=decision.route)
            output = self.text_small(req.payload)
            # text_small MUST return a dict with label and calibrated confidence.
            if not isinstance(output, dict) or 'confidence' not in output:
                raise ValueError('text_small must return {label, confidence}; confidence must be validated')
            confidence = float(output['confidence'])
            if not 0 <= confidence <= 1:
                raise ValueError('confidence out of range')
            log('verify', confidence=confidence)
            if confidence < self.confidence_threshold:
                log('escalate', route='large', reason='small-model confidence below threshold')
                output = self.text_large(req.payload)
                decision = Decision('small-then-large', 'Small model uncertain; executed fallback')
        elif req.modality == Modality.IMAGE:
            if req.quality_mode == 'preview':
                decision = Decision('fast', 'Preview requested; use few-step generation')
                log('select', route='fast')
                output = self.image_fast(req.payload)
            else:
                decision = Decision('quality', 'Quality mode requested; use reference pipeline')
                log('select', route='quality')
                output = self.image_quality(req.payload)
            # Automatic image-quality gate is a future phase; do not claim verification here.
            log('generated', quality_check='not_implemented')
        else:
            raise ValueError(f'Unsupported modality: {req.modality}')
        log('complete')
        return Trace(request={**asdict(req), 'modality': req.modality.value},
                     decision=asdict(decision), events=events, result=output,
                     total_latency_ms=round((perf_counter()-start)*1000,2))
