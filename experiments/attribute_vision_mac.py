"""Local MPS Qwen2.5-VL attribute verification on the 16 image development set.
This is diagnostic, not an independent quality claim.
"""
from __future__ import annotations
import argparse, csv, io, json, re, time, zipfile, gc
from pathlib import Path
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

def truth(value):
    return str(value).strip().lower() in ("true", "yes", "1", "1.0")

def load_records(csv_path):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows=list(csv.DictReader(f))
    required={"prompt_id","prompt","human_fast_object_present","human_standard_object_present","fast_s","standard_s"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV lacks required columns: {sorted(required)}")
    if any(not r["human_fast_object_present"] or not r["human_standard_object_present"] for r in rows):
        raise ValueError("Complete human labels before running")
    target=next((r for r in rows if str(r["prompt_id"])=="9"), None)
    if target and truth(target["human_fast_object_present"]):
        raise ValueError("Prompt 9 is still marked TRUE. Use the corrected CSV.")
    return rows

def load_image(archive, idx, size=448):
    filename=f"fast_{int(idx):02d}.png"
    paths=[n for n in archive.namelist() if n.endswith("/"+filename) or n==filename]
    if len(paths)!=1:
        raise ValueError(f"Expected exactly one {filename}; found {len(paths)}")
    image=Image.open(io.BytesIO(archive.read(paths[0]))).convert("RGB")
    image.thumbnail((size,size))
    return image

def decode_verdict(raw):
    try:
        match=re.search(r"\{.*?\}",raw,re.S)
        payload=json.loads(match.group()) if match else {}
    except (ValueError,AttributeError):
        payload={}
    verdict=str(payload.get("verdict","UNCERTAIN")).strip().upper()
    if verdict not in {"PASS","FAIL","UNCERTAIN"}: verdict="UNCERTAIN"
    return verdict,str(payload.get("observed_evidence",""))[:350],str(payload.get("missing_requirement",""))[:350]

def summarize(rows, records):
    by_id={str(k):v for k,v in records.items() if not v.get("error")}
    false_accept=unnecessary=success=escalations=0
    elapsed=0.0
    used=0
    for r in rows:
        x=by_id.get(str(r["prompt_id"]))
        if not x: continue
        used+=1
        escalate=x["verdict"]!="PASS"
        fast=truth(r["human_fast_object_present"])
        std=truth(r["human_standard_object_present"])
        false_accept+=int(not escalate and not fast)
        unnecessary+=int(escalate and fast)
        success+=int(std if escalate else fast)
        escalations+=int(escalate)
        elapsed+=float(r["fast_s"])+float(x["checker_s"])+(float(r["standard_s"]) if escalate else 0.0)
    return {"scope":"development_only","evaluated":used,"successes":success,"false_accepts":false_accept,
            "unnecessary_fallbacks":unnecessary,"fallbacks":escalations,
            "estimated_latency_seconds":round(elapsed,3),
            "estimated_latency_seconds_per_success":round(elapsed/success,3) if success else None,
            "note":"Uses recorded generation timings + measured MPS checker timing; NOT end-to-end inference."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",default="results/nimbo_image_v2_results.zip")
    p.add_argument("--labels",default="results/benchmark_v2_raw_labeled.csv")
    p.add_argument("--output",default="results/attribute_verdicts_mac.json")
    p.add_argument("--limit",type=int,default=16,help="Use 2 first to check memory")
    p.add_argument("--image-size",type=int,default=448)
    p.add_argument("--max-new-tokens",type=int,default=100)
    args=p.parse_args()
    if args.limit < 1 or args.image_size<224: p.error("limit>=1 and image-size>=224 required")
    labels=load_records(args.labels)
    with zipfile.ZipFile(args.images) as z:
        # Verify the input data *before* loading the 3B model.
        for r in labels[:args.limit]: load_image(z,r["prompt_id"],args.image_size)

    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS unavailable. This script intentionally avoids slow CPU mode.")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print("Loading",MODEL_ID,"onto MPS. Memory-conscious float16; avoid other large apps.")
    processor=AutoProcessor.from_pretrained(MODEL_ID)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID,dtype=torch.float16,low_cpu_mem_usage=True).to("mps").eval()
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    records={}
    if destination.exists():
        previous=json.loads(destination.read_text(encoding="utf-8"))
        records=previous.get("records",{}) if isinstance(previous,dict) else {}

    with zipfile.ZipFile(args.images) as z:
        for r in labels[:args.limit]:
            key=str(r["prompt_id"])
            if key in records and not records[key].get("error"):
                print("Skipping completed",key)
                continue
            image=load_image(z,key,args.image_size)
            instruction=("Assess whether this generated image clearly satisfies the original request. "
                "Check the required object, colors, and relationships. Do not assume unseen objects exist. "
                "Answer only JSON with verdict (PASS, FAIL, UNCERTAIN), observed_evidence, missing_requirement. "
                "If unclear choose UNCERTAIN. Request: "+r["prompt"])
            messages=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":instruction}]}]
            try:
                rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                inputs=processor(text=[rendered],images=[image],padding=True,return_tensors="pt")
                inputs={k:v.to("mps") for k,v in inputs.items()}
                torch.mps.synchronize()
                begin=time.perf_counter()
                with torch.inference_mode():
                    ids=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,
                                       pad_token_id=processor.tokenizer.eos_token_id)
                torch.mps.synchronize()
                duration=time.perf_counter()-begin
                raw=processor.batch_decode(ids[:,inputs["input_ids"].shape[-1]:],skip_special_tokens=True)[0]
                verdict,observed,missing=decode_verdict(raw)
                records[key]={"verdict":verdict,"checker_s":round(duration,4),
                        "observed_evidence":observed,"missing_requirement":missing,"raw":raw[:1000]}
                print(key,r.get("required_object",""),verdict,round(duration,2),"s")
                del inputs,ids
                gc.collect()
                torch.mps.empty_cache()
            except (RuntimeError,ValueError) as exc:
                records[key]={"error":str(exc)[:400]}
                print("Failed",key,records[key]["error"])
                if "out of memory" in str(exc).lower(): print("Stop other apps; retry with --image-size 336")
            destination.write_text(json.dumps({"model":MODEL_ID,"device":"mps","scope":"development_only",
                "records":records,"metrics":summarize(labels,records)},ensure_ascii=False,indent=2),encoding="utf-8")
    print("Saved",destination)
    print(json.dumps(summarize(labels,records),indent=2))
if __name__=="__main__":
    main()
