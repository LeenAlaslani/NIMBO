"""Measure real Qwen3-0.6B prefill+decode on the same prompts, with/without context selection.
Does not automatically claim response quality. Human review CSV is generated.
"""
import argparse,csv,json,time
from pathlib import Path
from nimbo.context_optimizer import select_context

TASKS=[
 {'id':'01','question':'What is the refund deadline and the fee?','context':('Company history: Founded in 2010. The team has 200 staff. The office is on River Street. Operations include sales and repairs. Support runs weekdays. The logo is blue. ' * 8)+'Refund requests must be submitted within 14 days. A processing fee of $5 applies. '+'Press releases are published quarterly. The website was redesigned recently. '*8,'required':['14 days','$5'],'budget':120},
 {'id':'02','question':'When is maintenance scheduled and who is on call?','context':('The engineering team ships weekly releases. Documentation lives in an internal wiki. The service has dashboards and alerts. '*12)+'Scheduled maintenance is Sunday at 02:00 UTC. The on-call owner is Maya. '+'Training materials are updated monthly. '*9,'required':['Sunday at 02:00 UTC','Maya'],'budget':130},
 {'id':'03','question':'ما موعد التسليم وما الحد الأعلى للميزانية؟','context':('الشركة لديها عدة فرق تعمل على منتجات مختلفة. قسم التسويق يراجع التصاميم. الاجتماعات الأسبوعية تعقد عن بعد. '*12)+'موعد التسليم هو 15 نوفمبر. الحد الأعلى للميزانية 8000 ريال. '+'التقارير يتم أرشفتها شهريًا. '*9,'required':['15 نوفمبر','8000 ريال'],'budget':140},
 {'id':'04','question':'What must be checked before releasing the update?','context':('The product supports laptops and tablets. Designers prefer small buttons. User feedback is collected weekly. '*12)+'Before release, verify data backups and run the security tests. '+'The roadmap includes a dark theme. '*10,'required':['data backups','security tests'],'budget':140},
]

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--dry-run',action='store_true',help='Token counts only: no model generation')
 ap.add_argument('--max-new-tokens',type=int,default=90)
 ap.add_argument('--output',default='results/context_token_v1')
 args=ap.parse_args()
 from transformers import AutoTokenizer
 name='Qwen/Qwen3-0.6B';tok=AutoTokenizer.from_pretrained(name)
 device=None;model=None
 if not args.dry_run:
  import torch
  from transformers import AutoModelForCausalLM
  device='mps' if torch.backends.mps.is_available() else 'cpu'
  model=AutoModelForCausalLM.from_pretrained(name,torch_dtype='auto').to(device).eval()
 def prompt(task,ctx):
  return f"Answer ONLY using the supplied context. If the answer is not stated, say 'Not in context'. Be concise.\nContext:\n{ctx}\nQuestion: {task['question']}"
 def render(s):
  return tok.apply_chat_template([{'role':'user','content':s}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
 records=[]
 for t in TASKS:
  selection=select_context(t['question'],t['context'],tok,t['budget'],t['required'])
  for mode,ctx in [('baseline',t['context']),('optimized',selection.context)]:
   rendered=render(prompt(t,ctx));input_tokens=len(tok.encode(rendered,add_special_tokens=False))
   answer='';duration=None;output_tokens=None
   if model:
    import torch
    ins=tok(rendered,return_tensors='pt').to(device)
    if device=='mps':torch.mps.synchronize()
    start=time.perf_counter()
    with torch.inference_mode(): output=model.generate(**ins,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
    if device=='mps':torch.mps.synchronize()
    duration=round(time.perf_counter()-start,4)
    tokens=output[0,ins['input_ids'].shape[1]:]
    output_tokens=int(tokens.numel());answer=tok.decode(tokens,skip_special_tokens=True).strip()
   records.append({'task_id':t['id'],'mode':mode,'selection_status':selection.status,
     'input_tokens':input_tokens,'output_tokens':output_tokens,'generation_s':duration,'answer':answer,
     'required_evidence_retained':selection.retained_required,'human_answer_correct':'',
     'question':t['question'],'required_answer_fragments':' | '.join(t['required'])})
  b,o=records[-2:]
  print(t['id'],selection.status,'tokens',b['input_tokens'],'->',o['input_tokens'],
        'seconds',b['generation_s'],'->',o['generation_s'])
 csvpath=out.with_suffix('.csv')
 with csvpath.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 summary={'model':name,'device':device,'dry_run':args.dry_run,'note':'Measured input tokens and optional local inference; manual answer verification still required.',
 'tasks':len(TASKS),'baseline_input_tokens':sum(r['input_tokens'] for r in records if r['mode']=='baseline'),
 'optimized_input_tokens':sum(r['input_tokens'] for r in records if r['mode']=='optimized'),
 'baseline_generation_s':round(sum(r['generation_s'] or 0 for r in records if r['mode']=='baseline'),4) if model else None,
 'optimized_generation_s':round(sum(r['generation_s'] or 0 for r in records if r['mode']=='optimized'),4) if model else None,
 'review_needed':'Fill human_answer_correct in CSV separately for each baseline and optimized answer; do not treat required fact retention as answer quality.'}
 out.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
 print('Files:',csvpath,out.with_suffix('.json'))
if __name__=='__main__': main()
