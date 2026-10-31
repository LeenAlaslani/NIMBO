"""Real local Qwen inference baseline. Not yet an adaptive router.
Run: python -m experiments.qwen_generate --prompt 'اشرح الذكاء الاصطناعي باختصار'
Needs: pip install 'transformers>=4.51,<5' torch
"""
import argparse, json, time

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-0.6B')
    p.add_argument('--prompt', default='اشرح الذكاء الاصطناعي في جملتين بسيطتين.')
    p.add_argument('--max-new-tokens', type=int, default=96)
    args = p.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if torch.cuda.is_available(): device='cuda'
    elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): device='mps'
    else: device='cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype='auto').to(device).eval()
    # Disable thinking for the first controlled baseline if the tokenizer supports it.
    messages=[{'role':'user','content':args.prompt}]
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors='pt')
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    ids = ids.to(device)
    def sync():
        if device=='cuda': torch.cuda.synchronize()
        elif device=='mps': torch.mps.synchronize()
    with torch.inference_mode():
        # Warmup is separate from reported latency.
        model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        sync(); start=time.perf_counter()
        out=model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        sync(); elapsed=time.perf_counter()-start
    new_tokens=out.shape[1]-ids.shape[1]
    answer=tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(json.dumps({'model':args.model,'device':device,'input_tokens':ids.shape[1], 'output_tokens':new_tokens,'generation_latency_s':round(elapsed,3),'tokens_per_second':round(new_tokens/elapsed,2) if elapsed else None,'answer':answer},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
