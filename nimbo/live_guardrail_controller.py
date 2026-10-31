"""Evidence-aware, bounded retry controller. Not a general truth verifier."""
from nimbo.quality_latency_guardrails import verify
from nimbo.context_answer_guard_v23 import answer_guard

def check_answer(question, evidence, answer):
    semantic=verify(question,evidence,answer)
    surface=answer_guard(question,evidence,answer)
    if semantic.status=='RETRY' or surface.status=='NEEDS_RETRY':
        return 'RETRY', semantic.reason if semantic.status=='RETRY' else surface.reason,semantic.expected
    if semantic.status=='SUPPORTED_BY_RULE' and surface.status!='NEEDS_REVIEW':
        return 'RULE_SUPPORTED',semantic.reason,semantic.expected
    return 'UNVERIFIED',surface.reason if surface.status=='NEEDS_REVIEW' else semantic.reason,''

def next_action(status,attempt,max_attempts,elapsed_s,budget_s,estimated_retry_s):
    if status!='RETRY':return 'STOP'
    if attempt>=max_attempts:return 'STOP_ATTEMPTS'
    if elapsed_s+estimated_retry_s>budget_s:return 'STOP_BUDGET'
    return 'RETRY'

def retry_guidance(question,expected):
    # Only derive expected entities from evidence-based rule extraction, never human labels.
    base='Answer exactly the question and use all evidence lines. Give the final person or value, not an intermediate entity. No placeholders or citation markers.'
    if expected:
        base+=' In the evidence, the final approver is '+expected+'. State this entity only if supported by the evidence.'
    return base+'\nQuestion: '+question
