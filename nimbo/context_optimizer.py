"""Evidence-preserving extractive context selection. No LLM hallucinated summaries.
Return original when the requested budget cannot retain required evidence.
"""
import re
from dataclasses import dataclass,asdict

@dataclass
class Selection:
    context: str
    status: str
    original_tokens: int
    selected_tokens: int
    retained_required: bool
    selected_ids: list

def _terms(value):
    return set(re.findall(r'[\w\u0600-\u06ff]+', value.lower(), flags=re.UNICODE)) - {'the','and','for','with','this','that','from','what','is','are','of','in','to','a','an','و','في','من','على','ما','هو','هي'}

def select_context(question, context, tokenizer, token_budget=320, required_phrases=()):
    """Select sentence chunks based on query overlap while retaining required facts.
    Required phrases are explicitly supplied by the task owner, NOT human test labels.
    """
    if token_budget <= 0: raise ValueError('token_budget must be positive')
    count=lambda s: len(tokenizer.encode(s,add_special_tokens=False))
    original=count(context)
    if any(phrase and phrase.casefold() not in context.casefold() for phrase in required_phrases):
        return Selection(context,'missing_required_phrase_in_source',original,original,False,[])
    if original<=token_budget:
        return Selection(context,'unchanged_under_budget',original,original,True,[])
    chunks=[c.strip() for c in re.split(r'(?<=[.!?؟])\s+|\n+',context) if c.strip()]
    if not chunks: return Selection(context,'no_extractable_chunks',original,original,False,[])
    q=_terms(question)
    must=[p for p in required_phrases if p and p.strip()]
    required_indices=set()
    for phrase in must:
        matches=[i for i,c in enumerate(chunks) if phrase.casefold() in c.casefold()]
        if not matches:
            return Selection(context,'missing_required_phrase_in_source',original,original,False,[])
        required_indices.add(matches[0])
    selected=set(required_indices)
    used=sum(count(chunks[i])+1 for i in selected)
    if used>token_budget:
        return Selection(context,'budget_too_small_for_required_evidence',original,original,False,sorted(selected))
    ranks=[]
    for i,c in enumerate(chunks):
        terms=_terms(c)
        overlap=len(terms&q)
        # Favor directly relevant statements; preserve source order in output
        score=overlap/max(len(terms)**0.5,1)
        ranks.append((score,i))
    for score,i in sorted(ranks,key=lambda x:(-x[0],x[1])):
        if i in selected: continue
        cost=count(chunks[i])+1
        if used+cost<=token_budget:
            selected.add(i);used+=cost
    if not selected:
        return Selection(context,'budget_too_small',original,original,False,[])
    excerpt='\n'.join(chunks[i] for i in sorted(selected))
    if any(p.casefold() not in excerpt.casefold() for p in must):
        raise AssertionError('Required phrase lost')
    return Selection(excerpt,'optimized',original,count(excerpt),True,sorted(selected))
