"""Qwen MPS live evaluation with limited retry; logs ALL attempts and cost.
Human reference criteria are not used to make runtime decisions.
"""
import argparse,csv,json,time
from pathlib import Path
from nimbo.context_optimizer_v2 import select_context_v2
from nimbo.context_answer_guard_v23 import answer_guard,prompt
from experiments.context_v2_diagnostics import CASES

MODEL='Qwen/Qwen3-0.6B'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--out',default='results/context_qwen_quality_v23')
    ap.add_argument('--limit',type=int,default=4)
    ap.add_argument('--max-retries',type=int,default=2,help='Up to two extra Qwen calls after first optimized attempt')
    args=ap.parse_args()
    if not 1<=args.limit<=4: ap.error('--limit must be 1..4')
    if args.max_retries not in (0,1,2): ap.error('--max-retries must be 0..2')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=None;torch=None
    if not args.dry_run:
        import torch as t
        from transformers import AutoModelForCausalLM
        if not t.backends.mps.is_available():raise RuntimeError('Requires Apple MPS. Use --dry-run for token counts.')
        torch=t
        model=AutoModelForCausalLM.from_pretrained(MODEL,dtype='auto').to('mps').eval()
    criteria={'refund':'Must explicitly state 14 days and $5',
              'missing':'Must decline to specify nonexistent warranty period',
              'conflict':'Must explicitly note two conflicting Friday times',
              'multihop':'Must answer Maya via Orion team'}
    rows=[]
    for name,q,ctx,required in CASES[:args.limit]:
        t0=time.perf_counter();selection=select_context_v2(q,ctx,tok,80,required);selection_s=time.perf_counter()-t0
        for mode in ('baseline','optimized'):
            blocked=mode=='optimized' and selection.status in {'MISSING_EVIDENCE','NEEDS_REVIEW'}
            if blocked:
                rows.append(dict(task_id=name,mode=mode,attempt=0,context_kind='abstain',selection_status=selection.status,
                   input_tokens=0,output_tokens=0,generation_s=0,selection_s=round(selection_s,6),
                   answer='Not in context: evidence missing.' if selection.status=='MISSING_EVIDENCE' else 'Needs review: conflicting sources.',
                   guard_status='ABSTAIN_REQUIRED',guard_reason=selection.status,review_criteria=criteria[name],human_correct=''))
                continue
            possible=[('full',ctx,False)] if mode=='baseline' else [('selected',selection.context,False),('selected_retry',selection.context,True),('full_retry',ctx,True)]
            possible=possible[:1] if mode=='baseline' else possible[:1+args.max_retries]
            for i,(kind,context,strict) in enumerate(possible):
                msg=prompt(q,context,strict)
                rendered=tok.apply_chat_template([{'role':'user','content':msg}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                inp_tokens=len(tok.encode(rendered,add_special_tokens=False))
                answer='';gen=0;out_tokens=0
                if model:
                    batch=tok(rendered,return_tensors='pt').to('mps')
                    torch.mps.synchronize();start=time.perf_counter()
                    with torch.inference_mode():
                        output=model.generate(**batch,max_new_tokens=112,do_sample=False,pad_token_id=tok.eos_token_id)
                    torch.mps.synchronize();gen=time.perf_counter()-start
                    fresh=output[0,batch['input_ids'].shape[-1]:]
                    out_tokens=int(fresh.numel());answer=tok.decode(fresh,skip_special_tokens=True).strip()
                guard=answer_guard(q,context,answer,selection.status if mode=='optimized' else 'UNCHANGED') if model else None
                status=guard.status if guard else 'NOT_EVALUATED'
                rows.append(dict(task_id=name,mode=mode,attempt=i+1,context_kind=kind,selection_status=selection.status,
                  input_tokens=inp_tokens,output_tokens=out_tokens,generation_s=round(gen,5),
                  selection_s=round(selection_s,6) if mode=='optimized' and i==0 else 0,
                  answer=answer,guard_status=status,guard_reason=guard.reason if guard else 'Dry run',
                  review_criteria=criteria[name],human_correct=''))
                print(name,mode,kind,status, 'tokens',inp_tokens,flush=True)
                if model and status!='NEEDS_RETRY':break
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary={'model':MODEL,'device':'mps' if model else None,'dry_run':args.dry_run,'tasks':args.limit,
       'attempts':len(rows),'retries':sum(r['attempt']>1 for r in rows),
       'baseline_input_tokens':sum(r['input_tokens'] for r in rows if r['mode']=='baseline'),
       'optimized_input_tokens_including_retries':sum(r['input_tokens'] for r in rows if r['mode']=='optimized'),
       'baseline_generation_s':round(sum(r['generation_s'] for r in rows if r['mode']=='baseline'),5),
       'optimized_generation_s_including_retries':round(sum(r['generation_s'] for r in rows if r['mode']=='optimized'),5),
       'optimized_selection_s':round(sum(r['selection_s'] for r in rows if r['mode']=='optimized'),5),
       'note':'Developer tasks only. Guard is NOT semantic verification. Human quality check required. All retries included in costs.'}
    dest.with_suffix('.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2));print('Saved',dest.with_suffix('.csv'),dest.with_suffix('.json'))
if __name__=='__main__':main()
