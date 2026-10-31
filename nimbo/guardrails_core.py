"""Task-agnostic, conservative quality and latency decision controller.

No human ground truth is consulted at runtime. Quality status comes from an
external verifier and is not interpreted as a universal correctness guarantee.
"""
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional
import math

class Action(str, Enum):
    ACCEPT = 'ACCEPT'
    RETRY = 'RETRY'
    ESCALATE = 'ESCALATE'
    UNVERIFIED = 'UNVERIFIED'

class Evidence(str, Enum):
    PASS = 'PASS'
    FAIL = 'FAIL'
    UNCERTAIN = 'UNCERTAIN'
    NOT_CHECKED = 'NOT_CHECKED'

SUPPORTED_TASKS = {'text_generation', 'text_classification', 'image_generation'}

@dataclass(frozen=True)
class Policy:
    task_type: str
    max_latency_s: float
    max_attempts: int = 2
    quality_floor: float = 0.90
    allow_escalation: bool = True
    min_next_attempt_s: float = 0.10

    def __post_init__(self):
        if self.task_type not in SUPPORTED_TASKS:
            raise ValueError('Unsupported task_type')
        if not (math.isfinite(self.max_latency_s) and self.max_latency_s > 0):
            raise ValueError('Invalid latency budget')
        if not (0 <= self.quality_floor <= 1):
            raise ValueError('Invalid quality floor')
        if self.max_attempts < 1 or not (math.isfinite(self.min_next_attempt_s) and self.min_next_attempt_s > 0):
            raise ValueError('Invalid attempts or next-attempt estimate')

@dataclass(frozen=True)
class Observation:
    evidence: Evidence
    elapsed_s: float
    attempts_used: int
    score: Optional[float] = None
    verifier_id: str = ''
    verifier_validated: bool = False
    escalation_available: bool = False
    already_escalated: bool = False

    def __post_init__(self):
        if not isinstance(self.evidence, Evidence):
            raise ValueError('evidence must be Evidence enum')
        if not (math.isfinite(self.elapsed_s) and self.elapsed_s >= 0):
            raise ValueError('Invalid elapsed')
        if self.attempts_used < 1:
            raise ValueError('attempts_used must be >= 1')
        if self.score is not None and not (math.isfinite(self.score) and 0 <= self.score <= 1):
            raise ValueError('Invalid score')

@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    remaining_s: float
    quality_certified: bool

def decide(policy: Policy, observation: Observation) -> Decision:
    """Decide what to do NEXT, not a claim that a model is globally correct.

    A PASS from an unvalidated verifier never becomes a certified ACCEPT.
    The time constraint is checked at decision boundaries, not a hard timeout.
    """
    remaining = max(0., policy.max_latency_s - observation.elapsed_s)
    if (observation.evidence is Evidence.PASS and observation.verifier_validated
            and observation.verifier_id and observation.score is not None
            and observation.score >= policy.quality_floor):
        if observation.elapsed_s <= policy.max_latency_s:
            return Decision(Action.ACCEPT, 'Validated verifier met policy floor within budget', remaining, True)
        return Decision(Action.UNVERIFIED, 'Latency budget already exceeded', remaining, False)

    if observation.attempts_used >= policy.max_attempts:
        return Decision(Action.UNVERIFIED, 'Attempt budget exhausted; quality not verified', remaining, False)
    if remaining < policy.min_next_attempt_s:
        return Decision(Action.UNVERIFIED, 'Not enough budget for another attempt', remaining, False)
    if (policy.allow_escalation and observation.escalation_available and not observation.already_escalated):
        return Decision(Action.ESCALATE, 'Quality not verified; use available stronger route', remaining, False)
    return Decision(Action.RETRY, 'Quality not verified; retry within policy budget', remaining, False)
