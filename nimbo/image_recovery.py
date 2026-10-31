"""Budgeted image recovery controller. Verification is an injected real callback.

No hidden access to human evaluation labels. Nothing is delivered as verified
unless the provided verifier actually returns PASS.
"""
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Callable, Any

class Verdict(str, Enum):
    PASS = 'PASS'
    FAIL = 'FAIL'
    UNCERTAIN = 'UNCERTAIN'

@dataclass
class Check:
    verdict: Verdict
    evidence: str = ''
    seconds: float = 0.0

@dataclass
class Attempt:
    stage: str
    verdict: str
    generation_seconds: float
    check_seconds: float
    evidence: str

@dataclass
class Outcome:
    status: str  # VERIFIED, UNVERIFIED, FAILED, BUDGET_EXHAUSTED
    image: Any = None
    attempts: list[Attempt] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    verified: bool = False

Generate = Callable[[str, int], Any]
Verify = Callable[[Any, str], Check]


def recover(prompt: str, generate: Generate, verify: Verify, *,
            budget_seconds: float = 12.0, max_attempts: int = 2,
            clock: Callable[[], float] = perf_counter) -> Outcome:
    """Try fast -> standard, then stop; a hard wall-clock timeout cannot be
    guaranteed for a blocking generator. Budget is checked BETWEEN calls.
    For real-time timeouts use worker process/service-level timeouts.
    """
    if max_attempts not in (1, 2):
        raise ValueError('Supported max_attempts: 1 (fast) or 2 (fast + standard)')
    if budget_seconds <= 0:
        raise ValueError('budget_seconds must be positive')
    start = clock()
    attempts = []
    last_image = None
    for stage in ('fast', 'standard')[:max_attempts]:
        if clock() - start >= budget_seconds:
            return Outcome('BUDGET_EXHAUSTED', None, attempts, clock()-start, False)
        t = clock()
        try:
            image = generate(stage, len(attempts))
        except Exception as exc:
            attempts.append(Attempt(stage, 'GENERATION_ERROR', clock()-t, 0.0, type(exc).__name__))
            continue
        gen_seconds = clock()-t
        last_image = image
        # Never mark image successful without calling the verifier.
        if clock() - start >= budget_seconds:
            attempts.append(Attempt(stage, 'NOT_VERIFIED_BUDGET', gen_seconds, 0.0, 'budget reached before verification'))
            return Outcome('BUDGET_EXHAUSTED', None, attempts, clock()-start, False)
        t = clock()
        try:
            result = verify(image, prompt)
            if not isinstance(result, Check) or not isinstance(result.verdict, Verdict):
                raise TypeError('verify must return Check(Verdict, evidence)')
        except Exception as exc:
            attempts.append(Attempt(stage, 'VERIFIER_ERROR', gen_seconds, clock()-t, type(exc).__name__))
            continue
        check_seconds = clock()-t
        attempts.append(Attempt(stage, result.verdict.value, gen_seconds, check_seconds, result.evidence))
        # Exceeded budget during blocking verifier: do not claim verified delivery.
        if clock()-start > budget_seconds:
            return Outcome('BUDGET_EXHAUSTED', None, attempts, clock()-start, False)
        if result.verdict == Verdict.PASS:
            return Outcome('VERIFIED', image, attempts, clock()-start, True)
    # Important: no automatic delivery of last_image after failed checks.
    return Outcome('FAILED', None, attempts, clock()-start, False)
