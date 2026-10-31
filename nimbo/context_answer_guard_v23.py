"""Surface-level evidence checks, not a factuality certificate."""
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class GuardResult:
    status:str
    reason:str

ABSTENTION=re.compile(r'\b(not in (?:the )?context|not provided|cannot determine|insufficient information|unknown)\b',re.I)
CITATION=re.compile(r'(?<!\w)\[(\d+)\]')
NUM=re.compile(r'(?<![\w])\d+(?:[.,:]\d+)*(?!\w)')

def answer_guard(question,evidence,answer,selection_status='OPTIMIZED'):
    if selection_status in {'MISSING_EVIDENCE','NEEDS_REVIEW'}:
        return GuardResult('ABSTAIN_REQUIRED','Missing or conflicting source evidence')
    if not evidence.strip() or not answer.strip():
        return GuardResult('NEEDS_RETRY','Missing evidence or answer')
    if ABSTENTION.search(answer):
        return GuardResult('NEEDS_RETRY','Model abstained despite retrieved evidence')
    citations=CITATION.findall(answer)
    cleaned=CITATION.sub('',answer)
    if citations:
        # e.g. fee is [2], a citation standing in for the value, not citing a factual value
        if re.search(r'\b(?:is|are|was|were|of|at|by|fee|deadline|cost|period)\s*[:=]?\s*\[\d+\]',answer,re.I):
            return GuardResult('NEEDS_RETRY','Citation placeholder used as answer value')
        source_line_count=sum(bool(x.strip()) for x in evidence.splitlines())
        if any(int(c)<1 or int(c)>source_line_count for c in citations):
            return GuardResult('NEEDS_REVIEW','Citation index out of range')
    actual_nums=set(NUM.findall(cleaned)); source_nums=set(NUM.findall(evidence))
    if actual_nums-source_nums:
        return GuardResult('NEEDS_RETRY','Ungrounded numerical statement: '+repr(sorted(actual_nums-source_nums)))
    return GuardResult('PENDING_HUMAN_QUALITY_REVIEW','Surface checks passed, semantic correctness not certified')

def prompt(question, context, stricter=False):
    directive=('Give the actual requested value(s) plainly in words/numbers before any citation. '
               'NEVER use a citation like [1] as a substitute for a value. '
               'For a multi-step question, connect all evidence lines. '
               'If evidence is absent answer exactly Not in context. '
               'No invented numbers, names or dates. ')
    if stricter: directive+='Check internally that every part of the question has an explicit answer. No citation brackets. '
    return (directive+'\nEvidence:\n'+ '\n'.join(f'[{i}] {line}' for i,line in enumerate(context.splitlines(),1) if line.strip())
            +'\nQuestion: '+question)
