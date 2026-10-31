"""Real-call orchestrator: dependency-injected image generators and checker.
A checker PASS is a model decision, never human-certified quality.
"""
from dataclasses import dataclass,asdict
from time import perf_counter
from typing import Callable,Any

@dataclass
class StageResult:
    stage:str
    prompt:str
    decision:str
    generation_s:float
    verification_s:float
    reason:str

@dataclass
class Result:
    status:str
    image:Any
    attempts:list[StageResult]
    elapsed_s:float
    checker_pass:bool
    # An image passed through an experimental heuristic; never claim verified correctness.

def repair_prompt(original:str,missing_object:str='',reason:str='')->str:
    """Deterministic, bounded prompt emphasis; no claims of learned optimality."""
    if not original.strip(): raise ValueError('A prompt is required')
    focus=(missing_object or '').strip()[:90]
    if focus:
        return (f'{original.strip()}. IMPORTANT: Clearly show {focus} as the unmistakable main subject, '
                'fully visible and recognizable with the required shape and attributes. '
                'Preserve the rest of the scene. Avoid replacing the specified object with lookalikes.')
    return (f'{original.strip()}. IMPORTANT: Preserve the required number of objects, colors, '
            'object shape and relationships exactly. Compose the central requested subject clearly.')

def run(prompt:str,generate:Callable[[str,str,int],Any],check:Callable[[Any,str],tuple],*,
        critical_object:str='',budget_s:float=40,max_attempts:int=3,clock=perf_counter)->Result:
    """Stages: fast, standard, repaired-standard. Blocking calls are not preempted.

    check(image,prompt) returns (PASS/FAIL/UNCERTAIN, reason). The checks run on
    outputs, never on human evaluation labels.
    """
    if not prompt.strip(): raise ValueError('empty prompt')
    if max_attempts not in (1,2,3): raise ValueError('max_attempts must be 1-3')
    if budget_s<=0: raise ValueError('budget_s must be >0')
    start=clock(); attempts=[]
    stages=('fast','standard','repaired_standard')[:max_attempts]
    last_reason=''
    for i,stage in enumerate(stages):
        if clock()-start>=budget_s:
            return Result('BUDGET_EXHAUSTED',None,attempts,clock()-start,False)
        current=prompt if stage!='repaired_standard' else repair_prompt(prompt,critical_object,last_reason)
        t=clock()
        try:
            im=generate(stage,current,i)
        except Exception as e:
            attempts.append(StageResult(stage,current,'GENERATION_ERROR',clock()-t,0,type(e).__name__))
            last_reason='generation error';continue
        gen_s=clock()-t
        if clock()-start>=budget_s:
            attempts.append(StageResult(stage,current,'BUDGET_BEFORE_CHECK',gen_s,0,'generator exceeded budget'))
            return Result('BUDGET_EXHAUSTED',None,attempts,clock()-start,False)
        t=clock()
        try:
            verdict,reason=check(im,current)
            if verdict not in ('PASS','FAIL','UNCERTAIN'):
                raise ValueError('checker returned invalid verdict')
        except Exception as e:
            verdict,reason='CHECKER_ERROR',type(e).__name__
        chk_s=clock()-t
        attempts.append(StageResult(stage,current,verdict,gen_s,chk_s,str(reason)[:400]))
        last_reason=str(reason)
        if clock()-start>=budget_s:
            return Result('BUDGET_EXHAUSTED',None,attempts,clock()-start,False)
        if verdict=='PASS':
            return Result('CHECKER_PASS_UNVALIDATED',im,attempts,clock()-start,True)
    return Result('FAILED_NO_DELIVERY',None,attempts,clock()-start,False)
