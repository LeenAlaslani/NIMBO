"""Conservative extractive retrieval. No test labels used for selection.
When evidence is ambiguous, return original context and flag for external verification.
"""
import re,time
from dataclasses import dataclass,asdict
STOP={'the','and','for','with','this','that','from','what','when','where','which','is','are','was','were','of','in','to','a','an','on','who','does','do','it','its','please','tell','me','و','في','من','على','ما','هو','هي','متى','مين','كم','هل','عن','الى','إلى'}
def terms(s): return set(re.findall(r'[\w\u0600-\u06ff]+',s.lower()))-STOP
def split_chunks(context): return [x.strip() for x in re.split(r'(?<=[.!?؟])\s+|\n+',context) if x.strip()]
def nums(s): return set(re.findall(r'\b\d+(?:[.,]\d+)?\b',s))
def numeric_fact_signature(s):
    """Only compare numerical claims with the same lexical subject/structure.
    Detects: 'Friday appointment is at 10:00' vs '...11:00'.
    Avoids false conflicts: 'deadline 14 days' vs 'fee $5'.
    This is NOT full semantic contradiction detection.
    """
    if not nums(s):
        return None
    normalized=re.sub(r'\b\d+(?:[.,:]\d+)*\b', '<number>',s.casefold())
    normalized=re.sub(r'[^\w<>\u0600-\u06ff]+',' ',normalized)
    return ' '.join(normalized.split())

@dataclass
class SelectionV2:
    context:str
    status:str
    input_tokens:int
    selected_tokens:int
    optimizer_ms:float
    selected_ids:list
    caution:str

def select_context_v2(question,context,tokenizer,token_budget=320,required_phrases=(),max_hops=2):
    if token_budget<1: raise ValueError('token_budget must be positive')
    start=time.perf_counter()
    count=lambda s:len(tokenizer.encode(s,add_special_tokens=False))
    orig=count(context)
    def result(value,status,indices=(),caution=''):
        return SelectionV2(value,status,orig,count(value),round((time.perf_counter()-start)*1000,3),list(indices),caution)
    required=[p.strip() for p in required_phrases if p.strip()]
    if any(p.casefold() not in context.casefold() for p in required):
        return result(context,'MISSING_EVIDENCE',caution='Caller-supplied required phrase absent; never invent it')
    if orig<=token_budget:return result(context,'UNCHANGED')
    chunks=split_chunks(context)
    if not chunks:return result(context,'NEEDS_REVIEW',caution='No extractable chunks')
    q=terms(question)
    relevance=[len(q & terms(c)) for c in chunks]
    if max(relevance,default=0)==0:
        return result(context,'NEEDS_REVIEW',caution='No lexical evidence for query; not safe to compress')
    # Cautious with near-duplicates: newest/oldest or version conflict requires disambiguation.
    # Without reliable structured timestamps, do not guess which conflicting fact is current.
    top=max(relevance)
    anchors=[i for i,v in enumerate(relevance) if v==top]
    if len(anchors)>1:
        # Different numerical facts only conflict if their underlying statement
        # is the same apart from the numerical values.
        by_subject={}
        for i in anchors:
            signature=numeric_fact_signature(chunks[i])
            if signature:
                by_subject.setdefault(signature,set()).add(tuple(sorted(nums(chunks[i]))))
        if any(len(values)>1 for values in by_subject.values()):
            return result(context,'NEEDS_REVIEW',caution='Conflicting values for the same numeric statement')
    must=set()
    for phrase in required:
        matches=[i for i,c in enumerate(chunks) if phrase.casefold() in c.casefold()]
        if not matches:return result(context,'NEEDS_REVIEW',caution='Required phrase crosses extraction boundary')
        must.add(matches[0])
    # Seed all directly relevant chunks instead of keeping only highest hit; preserve potentially multi-hop facts.
    eligible={i for i,v in enumerate(relevance) if v>0}
    eligible |=must
    # Link relevant chunks with other snippets sharing rare named tokens, bounded by hops.
    frontier=set(eligible)
    for _ in range(max_hops-1):
        new=set()
        for i in frontier:
            t=terms(chunks[i]); connectors={x for x in t if len(x)>=4 and x not in q}
            for j,c in enumerate(chunks):
                if j not in eligible and connectors & terms(c): new.add(j)
        eligible|=new;frontier=new
        if not frontier:break
    # Required chunks and direct matches first, then graph expansions; stable source-order in output.
    ranked=sorted(eligible,key=lambda i:(i not in must,-relevance[i],i))
    selected=[]; used=0
    for i in ranked:
        cost=count(chunks[i])+1
        if used+cost<=token_budget:selected.append(i);used+=cost
        elif i in must:
            return result(context,'BUDGET_UNSAFE',caution='Required evidence does not fit budget')
    if not selected:return result(context,'BUDGET_UNSAFE',caution='No evidence fits token budget')
    if any(i not in selected for i in must):return result(context,'BUDGET_UNSAFE')
    # If direct evidence can't fit, fail closed (no context optimization).
    if not any(relevance[i]>0 for i in selected):return result(context,'BUDGET_UNSAFE')
    text='\n'.join(chunks[i] for i in sorted(selected))
    if count(text)>=orig:return result(context,'UNCHANGED')
    return result(text,'OPTIMIZED',sorted(selected),caution='Extractive retrieval; semantic completeness not guaranteed')
