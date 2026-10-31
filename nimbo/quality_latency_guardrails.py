"""Bounded evidence-graph checks for a narrow class of approval questions.

This is *not* general natural-language entailment. Unrecognized structures stay
UNVERIFIED and must not be reported as verified facts.
"""
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str
    expected: str = ""

def _norm(text):
    return re.sub(r'\s+',' ',text.strip()).casefold()

def linked_approver(question,evidence):
    """Extract the target through two explicit relations: project -> team -> approver."""
    if not re.search(r'\b(who|whom)\b.*\bapprov',question,re.I):
        return None
    project_match=re.search(r'\b(?:project)\s+([A-Za-z][\w-]*)\b',question,re.I)
    if not project_match: return None
    project=re.escape(project_match.group(1))
    team_match=re.search(r'\bProject\s+'+project+r'\s+is\s+owned\s+by\s+([A-Za-z][\w-]*(?:\s+team)?)\b',evidence,re.I)
    if not team_match: return None
    team=team_match.group(1)
    # Explicit, evidence-anchored relations only. "Orion team approvals go through Maya."
    pattern=r'\b'+re.escape(team)+r'\s+approvals\s+go\s+through\s+([A-Za-z][\w-]*)\b'
    person=re.search(pattern,evidence,re.I)
    return (team,person.group(1)) if person else None

def verify(question,evidence,answer):
    answer=answer.strip()
    if not answer:
        return Verdict('RETRY','Empty answer')
    if re.search(r'\b(not in (?:the )?context|no evidence|cannot determine)\b',answer,re.I):
        if linked_approver(question,evidence):
            return Verdict('RETRY','Evidence exists but model abstained')
        return Verdict('UNVERIFIED','Model abstained; lack of evidence not established')
    link=linked_approver(question,evidence)
    if link:
        team,person=link
        answer_norm=_norm(answer)
        if re.search(r'\b'+re.escape(person.casefold())+r'\b',answer_norm):
            return Verdict('SUPPORTED_BY_RULE','Explicit two-hop approver matched',person)
        # Answer uses intermediary team in place of required approver
        if re.search(r'\b'+re.escape(team.casefold())+r'\b',answer_norm):
            return Verdict('RETRY','Intermediate team named instead of approving person',person)
        return Verdict('RETRY','Required approver not named',person)
    # Do not treat arbitrary lexical matches or numbers as proof of factual correctness.
    return Verdict('UNVERIFIED','No supported verification pattern available')

def decision(verdict,attempt,max_attempts,remaining_s,estimated_next_s):
    """A retry is allowed only for a detected failure and available time budget."""
    if verdict.status!='RETRY': return 'STOP'
    if attempt>=max_attempts: return 'STOP_UNVERIFIED'
    if remaining_s < estimated_next_s: return 'STOP_BUDGET'
    return 'RETRY'
