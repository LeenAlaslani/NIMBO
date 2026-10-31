"""Actual Qwen3-0.6B MPS generation with bounded retry + attempt-level accounting."""
import argparse,csv,json,time
from pathlib import Path
from experiments.context_v2_diagnostics import CASES
from nimbo.context_optimizer_v2 import select_context_v2
from nimbo.context_answer_guard_v23 import prompt
from nimbo.live_guardrail_controller import check_answer,next_action,retry_guidance
MODEL='Qwen/Qwen3-0.6B'

def run():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results/live_guardrails_v2')
    ap.add_argument('--budget-s',type=float,default=12.0,help='Per-task measured generate budget, excluding model load')
    ap.add_argument('--retry-estimate-s',type=float,default=2.0)
    ap.add_argument('--max-attempts',type=int,default=2)
    ap.add_argument('--max-new-tokens',type=int,default=100)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.budget_s<=0 or args.max_attempts<1 or args.retry_estimate_s<0:ap.error('Invalid retry budget')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=None;torch=None
    if not args.dry_run:
        import torch as torch_module
        from transformers import AutoModelForCausalLM
        if not torch_module.backends.mps.is_available():raise RuntimeError('Requires Apple MPS (or use --dry-run).')
        torch=torch_module
        model=AutoModelForCausalLM.from_pretrained(MODEL,dtype='auto').to('mps').eval()
    logs=[];summaries=[]
    for name,q,ctx,required in CASES:
        sel_start=time.perf_counter();selection=select_context_v2(q,ctx,tok,80,required)
        selection_s=time.perf_counter()-sel_start
        if selection.status in ('MISSING_EVIDENCE','NEEDS_REVIEW'):
            summaries.append(dict(task=name,status=selection.status,attempts=0,total_input_tokens=0,total_output_tokens=0,generation_s=0.0,selection_s=round(selection_s,6),final_answer='',quality='ABSTAIN_POLICY'))
            print(name,'ABSTAIN',selection.status,flush=True)
            continue
        evidence=selection.context if selection.status=='OPTIMIZED' else ctx
        spent=0.0;tokens_in=0;tokens_out=0;final='';state='UNVERIFIED';attempted=0
        for i in range(1,args.max_attempts+1):
            if i>1 and spent+args.retry_estimate_s>args.budget_s:
                state='STOP_BUDGET';break
            if i==1: msg=prompt(q,evidence,False)
            else:
                from nimbo.quality_latency_guardrails import verify
                inferred=verify(q,evidence,final).expected
                msg=prompt(q,evidence,True)+'\n'+retry_guidance(q,inferred)
            rendered=tok.apply_chat_template([{'role':'user','content':msg}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            input_ids=tok(rendered,return_tensors='pt')
            n_in=int(input_ids['input_ids'].shape[-1]);tokens_in+=n_in
            text='';duration=0.0;n_out=0
            if model is not None:
                encoded=input_ids.to('mps');torch.mps.synchronize();start=time.perf_counter()
                with torch.inference_mode():
                    generated=model.generate(**encoded,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
                torch.mps.synchronize();duration=time.perf_counter()-start
                fresh=generated[0,encoded['input_ids'].shape[-1]:];n_out=int(fresh.numel());text=tok.decode(fresh,skip_special_tokens=True).strip()
            tokens_out+=n_out;spent+=duration;attempted=i;final=text
            verdict,reason,expected=check_answer(q,evidence,text) if model else ('NOT_EVALUATED','dry run','')
            logs.append(dict(task=name,attempt=i,context_type=selection.status,input_tokens=n_in,output_tokens=n_out,generation_s=round(duration,5),answer=text,verdict=verdict,reason=reason,expected_rule_entity=expected))
            print(name,'attempt',i,verdict,round(duration,3),'s',flush=True)
            if verdict!='RETRY':state=verdict;break
            action=next_action(verdict,i,args.max_attempts,spent,args.budget_s,args.retry_estimate_s)
            if action!='RETRY':state=action;break
        summaries.append(dict(task=name,status=state,attempts=attempted,total_input_tokens=tokens_in,total_output_tokens=tokens_out,generation_s=round(spent,5),selection_s=round(selection_s,6),final_answer=final,quality='RULE_MATCH_ONLY' if state=='RULE_SUPPORTED' else 'NOT_CERTIFIED'))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as f:
        if logs:
            w=csv.DictWriter(f,fieldnames=list(logs[0]));w.writeheader();w.writerows(logs)
        else:f.write('task,attempt,answer\n')
    report=dict(model=MODEL,device='mps' if model is not None else None,dry_run=args.dry_run,budget_s=args.budget_s,max_attempts=args.max_attempts,
        tasks=summaries,total_attempts=sum(s['attempts'] for s in summaries),total_input_tokens=sum(s['total_input_tokens'] for s in summaries),total_output_tokens=sum(s['total_output_tokens'] for s in summaries),total_generation_s=round(sum(s['generation_s'] for s in summaries),5),total_selection_s=round(sum(s['selection_s'] for s in summaries),5),
        caveats=['Live latency measures generation only; model load excluded; budget checked between complete generation calls, not a hard interruption of one call.','RULE_SUPPORTED means matching one explicitly supported entity pattern, not general semantic correctness.','Missing/conflict abstentions consume no model generation but must be accounted as task outcomes.','No independent quality evidence or savings claim from four development tasks.'])
    out.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved',out.with_suffix('.json'),out.with_suffix('.csv'))
if __name__=='__main__':run()
