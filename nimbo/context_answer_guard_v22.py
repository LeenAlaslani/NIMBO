"""Conservative evidence guard for Qwen context answers.
It does not certify semantic correctness; ambiguous answers require human review.
"""
import re
from dataclasses import dataclass

@dataclass
class GuardResult:
    status: str
    reason: str

ABSTENTION=re.compile(r'\b(not in (?:the )?context|not provided|cannot determine|insufficient information|unknown)\b',re.I)

def check_answer(question, evidence, answer, selection_status):
    if selection_status in {'MISSING_EVIDENCE','NEEDS_REVIEW'}:
        return GuardResult('ABSTAIN_REQUIRED', 'Retrieval identified missing or conflicting evidence')
    if not evidence.strip():
        return GuardResult('NEEDS_REVIEW', 'No evidence text')
    if not answer.strip():
        return GuardResult('NEEDS_REVIEW', 'Empty model answer')
    if ABSTENTION.search(answer):
        return GuardResult('NEEDS_REVIEW', 'Model abstained; confirm whether evidence is sufficient')
    numbers=set(re.findall(r'\b\d+(?:\.\d+)?\b',answer))
    evidence_numbers=set(re.findall(r'\b\d+(?:\.\d+)?\b',evidence))
    if numbers-evidence_numbers:
        return GuardResult('NEEDS_REVIEW', 'Answer includes numbers not found in evidence')
    # Never claim a PASS from surface lexical matching alone.
    return GuardResult('PENDING_HUMAN_QUALITY_REVIEW', 'Automated checks cannot establish full factual correctness')


def evidence_instructions(question, context):
    return ("You must answer using only the evidence lines below. "
            "First connect facts from different lines when necessary: if A belongs to B and "
            "B's approvals go through C, then the approver for A is C. "
            "Do not demand an exact sentence match for the question. "
            "If the necessary link is absent, answer exactly: Not in context. "
            "Do not invent names, numbers, or dates. "
            "Give a short answer and cite the evidence line numbers used, e.g. [1][2].\n\n"
            "Evidence:\n"+"\n".join(f'[{i}] {s}' for i,s in enumerate(context.splitlines(),1) if s.strip())+
            "\nQuestion: "+question)
