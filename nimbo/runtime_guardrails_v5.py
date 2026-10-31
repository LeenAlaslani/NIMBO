"""Runtime guardrail decision + append-only metrics log.
Confidence is NOT accuracy; no generic factual verifier is available.
"""
from __future__ import annotations
import json, os, threading, time
from pathlib import Path

_LOCK=threading.Lock()
LOG=Path(os.environ.get("NIMBO_RUNTIME_EVIDENCE","results/runtime_evidence_v5.jsonl"))
LIMITS_MS={"classification":5000.0,"generation":12000.0,"image":180000.0}

def evaluate(workload, elapsed_ms, *, verified=False, verifier_id=None, reason=None):
    limit=LIMITS_MS.get(workload,12000.0)
    elapsed_ms=float(elapsed_ms)
    if elapsed_ms>limit:
        return {"action":"UNVERIFIED","reason":"Latency budget exceeded; completed inference cannot be interrupted retroactively",
                "quality_certified":False,"latency_budget_ms":limit,"latency_within_budget":False}
    if verified and verifier_id in {"human_verified","validated_task_checker"}:
        return {"action":"ACCEPT","reason":"Approved verifier supplied passing result",
                "quality_certified":True,"latency_budget_ms":limit,"latency_within_budget":True}
    return {"action":"UNVERIFIED","reason":reason or "No validated task-specific quality verifier is attached",
            "quality_certified":False,"latency_budget_ms":limit,"latency_within_budget":True}

def record(workload, route, report, *, status="completed"):
    """Store metadata only: no prompt, answer, or image."""
    guard=report["guardrail"]
    row={"timestamp_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
         "workload":workload,"route":route,"status":status,
         "action":guard["action"],"quality_certified":guard["quality_certified"],
         "latency_within_budget":guard["latency_within_budget"],
         "latency_budget_ms":guard["latency_budget_ms"],
         "end_to_end_ms":report.get("metrics",{}).get("end_to_end_ms"),
         "output_tokens":report.get("metrics",{}).get("output_tokens")}
    with _LOCK:
        LOG.parent.mkdir(parents=True,exist_ok=True)
        with LOG.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    return row

def recent(limit=25):
    if not LOG.exists():return []
    from collections import deque
    with _LOCK,LOG.open(encoding="utf-8") as f:
        lines=list(deque(f,maxlen=min(100,max(1,int(limit)))))
    return [json.loads(line) for line in lines if line.strip()]
