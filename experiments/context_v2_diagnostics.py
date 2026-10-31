"""CPU-only diagnostics, not an LLM accuracy benchmark."""
import argparse,json
from pathlib import Path
from nimbo.context_optimizer_v2 import select_context_v2
class BasicTokenizer:
    def encode(self,text,add_special_tokens=False):
        import re
        return re.findall(r"[\w\u0600-\u06ff]+|[^\w\s]",text)
CASES=[
 ('refund','What is the refund deadline and fee?', 'Old handbook covers hiring.\nRefund deadline is 14 days.\nRefund processing fee is $5.\nHR policy applies to employees.\n'+'Other unrelated operational note.\n'*30,('14 days','$5')),
 ('missing','What is the warranty period?', 'Refund deadline is 14 days.\n'+'Old operating procedures.\n'*30,('warranty',)),
 ('conflict','When is the appointment on Friday?', 'Friday appointment is at 10:00.\nFriday appointment is at 11:00.\n'+'Miscellaneous note.\n'*30,()),
 ('multihop','Who approves the release linked to Project Atlas?', 'Project Atlas is owned by Orion team.\nOrion team approvals go through Maya.\n'+'Unrelated note.\n'*30,())]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='results/context_v2_diagnostics.json');p.add_argument('--budget',type=int,default=80);args=p.parse_args()
 tok=BasicTokenizer();results=[]
 for name,q,c,must in CASES:
  v=select_context_v2(q,c,tok,args.budget,must)
  results.append(dict(case=name,question=q,status=v.status,original=v.input_tokens,selected=v.selected_tokens,optimizer_ms=v.optimizer_ms,selected_text=v.context[:1000],caution=v.caution))
 output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps({'mode':'cpu_diagnostics_only','tokenizer':'rough_regex_not_qwen','results':results},indent=2),encoding='utf-8')
 print(json.dumps({'mode':'cpu_diagnostics_only','results':[{k:x[k] for k in ('case','status','original','selected','optimizer_ms')} for x in results]},indent=2))
 print('Saved:',output)
if __name__=='__main__':main()
