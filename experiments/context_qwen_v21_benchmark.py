"""Development comparison of Qwen3-0.6B full vs v2.1 selected context.
Review CSV responses manually. All generation uses identical model and decoding config.
"""
import argparse,csv,json,time
from pathlib import Path
from nimbo.context_optimizer_v2 import select_context_v2
from experiments.context_v2_diagnostics import CASES

def build_tasks():
    return [{'id':name,'question':q,'context':ctx,'required':must,'budget':80,
             'expected_review':{'refund':'14 days AND $5','missing':'State warranty is not given',
                  'conflict':'Say two times conflict; do not choose one',
                  'multihop':'Maya (through Orion team)'}[name]}
            for name,q,ctx,must in CASES]

def choose_context(selection,original):
    # Safe policy: only OPTIMIZED may send shortened context.
    # NEEDS_REVIEW should not ask LLM to choose conflicting facts.
    if selection.status=='MISSING_EVIDENCE':return '', 'ABSTAIN_MISSING_EVIDENCE'
    if selection.status=='NEEDS_REVIEW':return '', 'ABSTAIN_NEEDS_REVIEW'
    if selection.status=='BUDGET_UNSAFE':return original,'FULL_CONTEXT_BUDGET_UNSAFE'
    return selection.context,selection.status

def render(tokenizer,question,context):
    prompt=("Use ONLY this context. If evidence is missing, say 'Not in context'. "
       "If there are conflicting facts, say they conflict and mention both. "
       "Never invent a fact. Answer concisely.\nContext:\n"+context+"\nQuestion: "+question)
    return tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--limit',type=int,default=4)
    ap.add_argument('--max-new-tokens',type=int,default=96)
    ap.add_argument('--out',default='results/context_qwen_v21')
    args=ap.parse_args()
    if not 1<=args.limit<=4:ap.error('limit must be 1..4')
    from transformers import AutoTokenizer
    name='Qwen/Qwen3-0.6B';tokenizer=AutoTokenizer.from_pretrained(name)
    torch=None;model=None;device=None
    if not args.dry_run:
        import torch as torch_module
        from transformers import AutoModelForCausalLM
        torch=torch_module
        if not torch.backends.mps.is_available():
            raise RuntimeError('This benchmark expects Mac MPS. Use --dry-run to check setup on CPU.')
        device='mps'
        model=AutoModelForCausalLM.from_pretrained(name,torch_dtype='auto').to(device).eval()
    results=[]
    for task in build_tasks()[:args.limit]:
        t0=time.perf_counter()
        selection=select_context_v2(task['question'],task['context'],tokenizer,task['budget'],task['required'])
        optimize_s=time.perf_counter()-t0
        optimized_ctx,optimized_status=choose_context(selection,task['context'])
        for mode in ('baseline','optimized'):
            ctx=task['context'] if mode=='baseline' else optimized_ctx
            abstain=(mode=='optimized' and optimized_status.startswith('ABSTAIN_'))
            # Abstention skips generation entirely. Report as such; do not compare its quality to model output automatically.
            rendered=render(tokenizer,task['question'],ctx) if not abstain else ''
            token_count=len(tokenizer.encode(rendered,add_special_tokens=False)) if not abstain else 0
            generation_s=None;output_tokens=0;answer='';status='ABSTAIN' if abstain else 'GENERATED'
            if abstain:
                answer={'ABSTAIN_MISSING_EVIDENCE':'Not in context: required evidence is missing.',
                        'ABSTAIN_NEEDS_REVIEW':'Needs review: the available evidence is conflicting or ambiguous.'}[optimized_status]
            elif model:
                inputs=tokenizer(rendered,return_tensors='pt').to(device)
                torch.mps.synchronize();t1=time.perf_counter()
                with torch.inference_mode():
                    outs=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
                torch.mps.synchronize();generation_s=time.perf_counter()-t1
                ids=outs[0,inputs['input_ids'].shape[-1]:]
                output_tokens=int(ids.numel());answer=tokenizer.decode(ids,skip_special_tokens=True).strip()
            results.append(dict(task_id=task['id'],mode=mode,selection_status=selection.status,
                execution_status=status,question=task['question'],input_tokens=token_count,
                output_tokens=output_tokens,generation_s=round(generation_s,5) if generation_s is not None else '',
                optimizer_ms=round(optimize_s*1000,4) if mode=='optimized' else 0,
                total_measured_s=round(generation_s+(optimize_s if mode=='optimized' else 0),5) if generation_s is not None else (round(optimize_s,5) if abstain else ''),
                answer=answer,expected_review=task['expected_review'],human_answer_correct=''))
        a,b=results[-2:]
        print(task['id'],selection.status,f"tokens: {a['input_tokens']} -> {b['input_tokens']}",
              f"generation: {a['generation_s']} -> {b['generation_s']}",flush=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    csvpath=out.with_suffix('.csv')
    with csvpath.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    def agg(mode,key):
        vals=[float(r[key]) for r in results if r['mode']==mode and r[key]!='']
        return round(sum(vals),5) if vals else None
    summary={'model':name,'device':device,'dry_run':args.dry_run,'tasks':args.limit,
             'policy':'OPTIMIZED => shortened context; missing/conflict => abstain; unsafe => original',
             'baseline_input_tokens':agg('baseline','input_tokens'),
             'optimized_input_tokens':agg('optimized','input_tokens'),
             'baseline_generation_s':agg('baseline','generation_s'),
             'optimized_generation_s':agg('optimized','generation_s'),
             'optimized_context_selection_s':round(sum(float(r['optimizer_ms']) for r in results if r['mode']=='optimized')/1000,6),
             'optimized_abstentions':sum(r['execution_status']=='ABSTAIN' for r in results),
             'warning':'DO NOT infer accuracy from token savings; human labels are required. Abstentions are not ordinary generated answers. Development tasks only.'}
    out.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));print('Saved',csvpath,out.with_suffix('.json'))
if __name__=='__main__':main()
