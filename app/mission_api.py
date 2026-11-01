"""Local NIMBO Mission Control API. Run: uvicorn app.mission_api:app --reload --port 8000"""
from __future__ import annotations
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from urllib.request import Request as URLRequest, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
import base64
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nimbo.text_backend import load_classifier
from nimbo.runtime_guardrails_v5 import evaluate as runtime_guardrail, record as record_evidence, recent as recent_evidence

app = FastAPI(title="NIMBO Mission API", version="0.4")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["GET", "POST"], allow_headers=["*"])

SMALL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
LARGE = "textattack/bert-base-uncased-SST-2"
POLICY = Path("results/rotten_tomatoes_v4.json")

class Task(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    workload: str = "classification"

@lru_cache(maxsize=2)
def model(name: str):
    return load_classifier(name, device=-1, sst2_labels=True)

def policy():
    if not POLICY.exists():
        raise HTTPException(503, "Policy report is missing: results/rotten_tomatoes_v4.json")
    report=json.loads(POLICY.read_text())
    chosen=report.get("selected_policy", {})
    kind=chosen.get("type", "small_only")
    threshold=chosen.get("threshold")
    if kind == "adaptive" and not isinstance(threshold, (int,float)):
        raise HTTPException(503, "Adaptive threshold missing from saved policy")
    return kind, threshold

@app.get("/api/health")
def health():
    return {"ok":True,"policy_available":POLICY.exists(),"workloads":{"classification":"ready","image":"connected_local_gpu" if probe_image_gateway_v22() else "not_connected","generation":"ready_on_first_load"}}

@app.post("/api/run")
def run(task: Task):
    if task.workload != "classification":
        raise HTTPException(501, "This workload is not connected yet; no simulated inference")
    kind,threshold=policy()
    trace=[]
    begun=time.perf_counter()
    def event(step, message, **details):
        trace.append({"step":step,"message":message,"at_ms":round((time.perf_counter()-begun)*1000,3),**details})
    event("analyze", "Sentiment classification detected")
    event("route", f"Loaded saved policy: {kind}", threshold=threshold)
    try:
        if kind == "large_only":
            t=time.perf_counter(); result=model(LARGE)(task.text); large_ms=(time.perf_counter()-t)*1000
            route="large_only"; small_ms=0.0
            event("execute", "Executed BERT", model=LARGE)
        else:
            t=time.perf_counter(); result=model(SMALL)(task.text); small_ms=(time.perf_counter()-t)*1000
            event("execute", "Executed DistilBERT", model=SMALL, confidence=result['confidence'])
            route="small_only"; large_ms=0.0
            if kind == "adaptive" and result["confidence"] < threshold:
                event("route", "Escalating: confidence is below the calibrated threshold")
                t=time.perf_counter(); result=model(LARGE)(task.text); large_ms=(time.perf_counter()-t)*1000
                event("execute", "Executed fallback BERT", model=LARGE)
                route="small_then_large"
        event("verify", "Prediction returned; confidence is not a guarantee of correctness")
        event("deliver", "Result delivered")
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {type(exc).__name__}: {exc}") from exc
    report={"workload":"classification", "result":result,"route":route,"policy":{"kind":kind,"threshold":threshold},"metrics":{"end_to_end_ms":round((time.perf_counter()-begun)*1000,2),"small_inference_ms":round(small_ms,2),"large_inference_ms":round(large_ms,2)},"trace":trace,"measurement_note":"Single request; cold start and model loading may be included. No baseline savings are inferred."}
    report["guardrail"]=runtime_guardrail("classification",report["metrics"]["end_to_end_ms"],reason="Classifier confidence is not ground-truth accuracy")
    trace.append({"step":"guardrail","message":report["guardrail"]["reason"],"action":report["guardrail"]["action"]})
    record_evidence("classification",route,report)
    return report


@app.post("/api/run/stream")
def run_stream(task: Task):
    """NDJSON events are emitted when the backend reaches each actual operation.

    The frontend can animate between events. No synthetic progress percentage.
    """
    if task.workload == "image":
        raise HTTPException(501, "GPU image inference is not connected")
    if task.workload == "generation":
        return StreamingResponse(generate_stream(task), media_type="application/x-ndjson",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
    kind, threshold = policy()
    def events():
        begun = time.perf_counter()
        def emit(stage, message, **details):
            return json.dumps({"type":"event", "stage":stage,"message":message,
                "at_ms":round((time.perf_counter()-begun)*1000,2),**details}) + "\n"
        yield emit("analyze","Sentiment classification task received")
        yield emit("route",f"Using calibrated policy: {kind}",policy=kind,threshold=threshold)
        try:
            small_ms=large_ms=0.0
            if kind == "large_only":
                yield emit("execute","Running BERT",model=LARGE)
                t=time.perf_counter();result=model(LARGE)(task.text);large_ms=(time.perf_counter()-t)*1000
                route="large_only"
            else:
                yield emit("execute","Running DistilBERT",model=SMALL)
                t=time.perf_counter();result=model(SMALL)(task.text);small_ms=(time.perf_counter()-t)*1000
                route="small_only"
                if kind=="adaptive" and result["confidence"] < threshold:
                    yield emit("route","Confidence below policy threshold; escalating",confidence=result["confidence"])
                    yield emit("execute","Running fallback BERT",model=LARGE)
                    t=time.perf_counter();result=model(LARGE)(task.text);large_ms=(time.perf_counter()-t)*1000
                    route="small_then_large"
            yield emit("verify","Inference finished. Model confidence is not accuracy verification")
            report={"workload":"classification","result":result,"route":route,
                "policy":{"kind":kind,"threshold":threshold},
                "metrics":{"end_to_end_ms":round((time.perf_counter()-begun)*1000,2),
                "small_inference_ms":round(small_ms,2),"large_inference_ms":round(large_ms,2)},
                "measurement_note":"Request latency may include cold model loading. No compute savings inferred."}
            report["guardrail"]=runtime_guardrail("classification",report["metrics"]["end_to_end_ms"],reason="Classifier confidence is not ground-truth accuracy")
            yield emit("guardrail",report["guardrail"]["reason"],action=report["guardrail"]["action"])
            record_evidence("classification",route,report)
            yield emit("deliver","Result ready; quality not certified")
            yield json.dumps({"type":"result","report":report})+"\n"
        except Exception as exc:
            yield json.dumps({"type":"error","message":f"{type(exc).__name__}: {exc}"})+"\n"
    return StreamingResponse(events(),media_type="application/x-ndjson",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@lru_cache(maxsize=1)
def qwen_engine():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name="Qwen/Qwen3-0.6B"
    tok=AutoTokenizer.from_pretrained(name)
    model=AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
    device="mps" if torch.backends.mps.is_available() else "cpu"
    model=model.to(device).eval()
    return tok,model,device

def generate_stream(task: Task):
    """Real local Qwen generation; streamed NDJSON milestones, not token streaming."""
    def records():
        from time import perf_counter
        import torch
        start=perf_counter()
        def emit(stage,message,**extra):
            return json.dumps({"type":"event","stage":stage,"message":message,"at_ms":round((perf_counter()-start)*1000,2),**extra})+"\n"
        yield emit("analyze","Text generation request received")
        yield emit("route","Using local Qwen3-0.6B for generation")
        yield emit("execute","Loading/running local Qwen model")
        try:
            tok,net,device=qwen_engine()
            messages=[{"role":"user","content":task.text}]
            input_text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tok(input_text,return_tensors="pt",return_attention_mask=True).to(device)
            with torch.inference_mode():
                output=net.generate(**inputs,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)
            new=output[0][inputs['input_ids'].shape[1]:]
            answer=tok.decode(new,skip_special_tokens=True).strip()
            yield emit("verify","Generated text returned; no automatic factuality guarantee")
            report={"workload":"generation","result":{"answer":answer},"route":"qwen3_0_6b","metrics":{"end_to_end_ms":round((perf_counter()-start)*1000,2),"output_tokens":len(new)},"measurement_note":"Local generation request including model cold load if applicable; no quality benchmark inferred."}
            report["guardrail"]=runtime_guardrail("generation",report["metrics"]["end_to_end_ms"],reason="Generated text has no task-specific evidence verifier")
            yield emit("guardrail",report["guardrail"]["reason"],action=report["guardrail"]["action"])
            record_evidence("generation",report["route"],report)
            yield emit("deliver","Text ready; quality not certified")
            yield json.dumps({"type":"result","report":report},ensure_ascii=False)+"\n"
        except Exception as exc:
            yield json.dumps({"type":"error","message":f"{type(exc).__name__}: {exc}"})+"\n"
    return records()


# The image gateway is optional and OFF by default. Only a locally managed
# service is allowed. Never send credentials or raw arbitrary URLs from users.
def image_gateway_url():
    address = os.getenv("NIMBO_IMAGE_GATEWAY", "").strip().rstrip("/")
    if not address:
        return None
    parsed = urlparse(address)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.username or parsed.password or parsed.query:
        raise HTTPException(503, "Image gateway must be a local HTTP endpoint (localhost or 127.0.0.1)")
    return address

@app.get("/api/registry")
def registry():
    return {"models": [
        {"id": SMALL, "workload": "classification", "runtime": "local CPU", "status": "ready_on_first_load"},
        {"id": LARGE, "workload": "classification", "runtime": "local CPU", "status": "ready_on_first_load"},
        {"id": "lvwerra/distilbert-imdb", "workload": "classification", "runtime": "local CPU / opt-in download", "status": "ready_on_first_load"},
        {"id": "Qwen/Qwen3-0.6B", "workload": "generation", "runtime": "local MPS/CPU", "status": "ready_on_first_load"},
        {"id": "stable-diffusion-v1-5 + LCM-LoRA", "workload": "image", "runtime": "external local GPU gateway",
         "status": "connected_local_gpu" if probe_image_gateway_v22() else "not_connected"}
    ], "image_gateway_connected": probe_image_gateway_v22()}

@app.post("/api/image")
def image_run(task: Task):
    url = image_gateway_url()
    if not url:
        raise HTTPException(503, "GPU image gateway is not configured. No simulated images.")
    started = time.perf_counter()
    payload = json.dumps({"prompt": task.text, "quality_policy": "verify_and_fallback"}).encode()
    request = URLRequest(url+"/generate", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read(12_000_000))
    except (HTTPError, URLError, TimeoutError, ValueError) as e:
        raise HTTPException(502, f"GPU gateway failed: {type(e).__name__}") from e
    # Contract: gateway returns real PNG as base64 plus real trace and metrics.
    if not isinstance(result, dict) or not isinstance(result.get("image_base64"), str):
        raise HTTPException(502, "Invalid GPU gateway response")
    try:
        binary = base64.b64decode(result["image_base64"], validate=True)
    except Exception as e:
        raise HTTPException(502, "Invalid image payload") from e
    if len(binary)>8_000_000 or not binary.startswith(b"\\x89PNG\\r\\n\\x1a\\n"):
        raise HTTPException(502, "Image gateway response must be a PNG under 8 MB")
    report={"workload":"image", "result":{"image_base64": result["image_base64"]},
            "route":result.get("route","gateway"),
            "metrics":{"end_to_end_ms":round((time.perf_counter()-started)*1000,2)},
            "trace": result.get("trace",[]),
            "measurement_note":"Includes local gateway network and actual inference; no unmeasured savings are claimed."}
    report["guardrail"]=runtime_guardrail("image",report["metrics"]["end_to_end_ms"],reason="Gateway image has no independently validated quality result")
    record_evidence("image",report["route"],report)
    return report


@app.get("/nimbo/evidence/recent")
def runtime_evidence(limit: int=25):
    return {"events":recent_evidence(limit),"type":"runtime_metadata_not_quality_benchmark"}

# NIMBO_COMPANY_WORKSPACE_V15_BACKEND
from app.nimbo_compare_v15 import router as nimbo_v15_router
app.include_router(nimbo_v15_router)

# NIMBO_CONNECTED_EVIDENCE_V16
from app.nimbo_connected_evidence_v16 import router as connected_evidence_router
app.include_router(connected_evidence_router)

# NIMBO_V20_SAME_MODEL_TRIAL: external model onboarding and same-model INT8 benchmark
from app.nimbo_optimize_v20 import router as same_model_router_v20
app.include_router(same_model_router_v20)

# NIMBO_V20_2_ONNX_TRIAL: paired FP32 CPU backend
from app.nimbo_onnx_v20 import router as onnx_optimization_router_v20
app.include_router(onnx_optimization_router_v20)

# NIMBO_V21_INDEPENDENT_VALIDATION: disjoint heldout ONNX validation; review-only candidate export
from app.nimbo_validate_v21 import router as onnx_heldout_validation_router_v21
app.include_router(onnx_heldout_validation_router_v21)

# NIMBO_V21_1_CONTROLLED_ADOPTION; isolated local opt-in only, production API remains unchanged.
from app.nimbo_adoption_v21_1 import router as adoption_router_v21_1
app.include_router(adoption_router_v21_1)

# NIMBO_IMAGE_GATEWAY_V22: real local GPU-only image routing and manual review
from app.nimbo_image_connector_v22 import router as image_router_v22, probe_image_gateway_v22
app.include_router(image_router_v22)

# NIMBO_ADAPTIVE_IMAGE_V24: sequential, human-gated GPU diffusion
from app.nimbo_adaptive_image_v24 import router as adaptive_image_router_v24
app.include_router(adaptive_image_router_v24)

# NIMBO_VERA_V25: real optional local image verifier (keeps v24 manual routes)
from app.nimbo_vera_v25_api import router as vera_v25_router
app.include_router(vera_v25_router)

# NIMBO_VERA_V26_AUTONOMOUS
from app.nimbo_vera_v26_api import router as vera_v26_router
app.include_router(vera_v26_router)

# NIMBO DEMO: local-only verified ONNX export and identity check; never activates uploaded files.
from app.nimbo_demo_artifact_v1 import router as nimbo_demo_artifact_router_v1
app.include_router(nimbo_demo_artifact_router_v1)

# Supported real model-file upload and actual isolated local ONNX evaluation.
from app.nimbo_company_model_v1 import router as company_model_pilot_router
app.include_router(company_model_pilot_router)

# NIMBO_M7A_AUTO_OPTIMIZER_LOCAL_ONLY
from app.nimbo_auto_jobs_api_v1 import router as nimbo_auto_optimizer_m7_router
app.include_router(nimbo_auto_optimizer_m7_router)
