"""Real Qwen3 0.6B MPS quality experiment on v2.1 development tasks.
Reports inference and guard outcomes; human evaluation is mandatory.
"""
import argparse,csv,json,time
from pathlib import Path
from nimbo.context_optimizer_v2 import select_context_v2
from nimbo.context_answer_guard_v22 import check_answer,evidence_instructions
from experiments.context_v2_diagnostics import CASES

def tasks():
    return [{'id':name,'question':q,'context':ctx,'required':must,'budget':80,
             'review_criteria':{'refund':'Both 14 days and $5 must be present',
             'missing':'Must decline to specify warranty period',
             'conflict':'Must flag conflicting Friday times, not choose either',
             'multihop':'Must identify Maya, connecting Atlas -> Orion -> Maya'}[name]}
            for name,q,ctx,must in CASES]

def run():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--out',default='results/context_qwen_quality_v22')
    ap.add_argument('--limit',type=int,default=4)
    args=ap.parse_args()
    if not 1<=args.limit<=4: ap.error('limit must be 1..4')
    from transformers import AutoTokenizer
    mid='Qwen/Qwen3-0.6B';tok=AutoTokenizer.from_pretrained(mid)
    model=None;torch=None
    if not args.dry_run:
        import torch as t
        from transformers import AutoModelForCausalLM
        if not t.backends.mps.is_available():
            raise RuntimeError('Requires Mac MPS for inference; --dry-run for tokenizer only')
        torch=t
        model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype='auto').to('mps').eval()
    rows=[]
    for task in tasks()[:args.limit]:
        start=time.perf_counter()
        selection=select_context_v2(task['question'],task['context'],tok,task['budget'],task['required'])
        selection_s=time.perf_counter()-start
        for mode in ('baseline','optimized'):
            context=task['context'] if mode=='baseline' else selection.context
            blocked=(mode=='optimized' and selection.status in ('MISSING_EVIDENCE','NEEDS_REVIEW'))
            # Use the exact same instruction formatting across both branches.
            content=evidence_instructions(task['question'],context)
            rendered=tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,
                                             add_generation_prompt=True,enable_thinking=False)
            tokens=len(tok.encode(rendered,add_special_tokens=False)) if not blocked else 0
            answer='';gen_s=None;out_tokens=0
            if blocked:
                answer=('Not in context: evidence missing.' if selection.status=='MISSING_EVIDENCE'
                        else 'Needs review: source claims conflict.')
            elif model:
                batch=tok(rendered,return_tensors='pt').to('mps')
                torch.mps.synchronize();t0=time.perf_counter()
                with torch.inference_mode():
                    output=model.generate(**batch,max_new_tokens=128,do_sample=False,
                                          pad_token_id=tok.eos_token_id)
                torch.mps.synchronize();gen_s=time.perf_counter()-t0
                fresh=output[0,batch['input_ids'].shape[-1]:]
                answer=tok.decode(fresh,skip_special_tokens=True).strip()
                out_tokens=int(fresh.numel())
            if blocked:
                gate='ABSTAIN_REQUIRED';reason='Retrieval status prevents generation'
            elif args.dry_run:
                gate='NOT_EVALUATED';reason='Dry run: no model output'
            else:
                g=check_answer(task['question'],context,answer,selection.status if mode=='optimized' else 'UNCHANGED')
                gate=g.status;reason=g.reason
            rows.append(dict(task_id=task['id'],mode=mode,question=task['question'],
                selection_status=selection.status,execution_status='ABSTAIN' if blocked else 'GENERATED',
                input_tokens=tokens,output_tokens=out_tokens,generation_s=round(gen_s,5) if gen_s is not None else '',
                selection_s=round(selection_s,6) if mode=='optimized' else 0,
                guard_status=gate,guard_reason=reason,answer=answer,
                review_criteria=task['review_criteria'],human_correct=''))
            print(task['id'],mode,selection.status,gate,'input tokens',tokens,flush=True)
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    totals=lambda mode,key:round(sum(float(r[key]) for r in rows if r['mode']==mode and r[key]!=''),5)
    summary={'model':mid,'device':'mps' if model else None,'dry_run':args.dry_run,
      'tasks':args.limit,'baseline_input_tokens':totals('baseline','input_tokens'),
      'optimized_input_tokens':totals('optimized','input_tokens'),
      'baseline_generation_s':totals('baseline','generation_s'),
      'optimized_generation_s':totals('optimized','generation_s'),
      'optimized_context_selection_s':totals('optimized','selection_s'),
      'optimized_abstentions':sum(r['mode']=='optimized' and r['execution_status']=='ABSTAIN' for r in rows),
      'quality_guard':'Conservative flagging ONLY; no automatic factual pass',
      'warning':'Four development tasks; human quality labeling mandatory. Abstentions separately tracked. Not held-out validation.'}
    target.with_suffix('.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2))
    print('Saved',target.with_suffix('.csv'),target.with_suffix('.json'))
if __name__=='__main__':run()
