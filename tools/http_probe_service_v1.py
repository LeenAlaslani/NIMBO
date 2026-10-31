"""Isolated, read-only loopback HTTP wrapper around NIMBO's staged CPU engines.

Not the production/adoption API, not a deployment; no candidate activation and no
production routing changes. Starts only on 127.0.0.1 via the companion CLI.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from tools.concurrency_probe_v1 import SOURCE_SELECTION, validate_lineage

ENGINE = None
META = None


class InferRequest(BaseModel):
    path: Literal['baseline', 'candidate']
    text: str = Field(min_length=1, max_length=3000)


def initialize_engine():
    """Require a pinned, previously accepted staged artifact; no adoption state writes."""
    from app.nimbo_adoption_v21_1 import _review, _prepare_cache
    from tools.staged_candidate_smoke_v1 import latest_exported_run

    selection = json.loads(SOURCE_SELECTION.read_text(encoding='utf8'))
    review = _review(latest_exported_run())  # verifies manifest and ONNX SHA256
    threads = validate_lineage(selection, review['validation'], review)
    eng = _prepare_cache(review)
    return eng, {
        'run_id': review['run_id'],
        'model_revision': review['revision'],
        'artifact_sha256': review['sha256'],
        'baseline_threads': review['validation']['thread_count'],
        'candidate_threads': threads,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE, META
    ENGINE, META = initialize_engine()
    try:
        yield
    finally:
        ENGINE, META = None, None


app = FastAPI(title='NIMBO local HTTP proof harness — NOT production', lifespan=lifespan)


def local_only(request: Request):
    address = request.client.host if request.client else None
    if address not in ('127.0.0.1', '::1', 'testclient'):
        raise HTTPException(status_code=403, detail='Loopback only')
    if request.headers.get('origin'):
        raise HTTPException(status_code=403, detail='Browser origins are not allowed')


@app.get('/health')
def health(request: Request):
    local_only(request)
    if ENGINE is None or META is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    return {**META, 'ready': True, 'production_deployed': False, 'adoption_state_changed': False,
            'service_kind': 'isolated_http_probe'}


def inference(eng, which: str, text: str):
    import numpy as np
    import torch

    t0 = time.perf_counter()
    encoded = eng['tokenizer'](text, max_length=256, truncation=True, return_tensors='np')
    inputs = {k: np.asarray(v, dtype=np.int64) for k, v in encoded.items() if k in eng['inputs']}
    if 'input_ids' not in inputs:
        raise RuntimeError('Missing input_ids')
    t1 = time.perf_counter()
    if which == 'baseline':
        with torch.inference_mode():
            scores = eng['baseline'](**{k: torch.from_numpy(v) for k, v in inputs.items()}).logits.detach().numpy()
    elif which == 'candidate':
        scores = eng['candidate'].run(None, inputs)[0]
    else:
        raise ValueError('Unsupported path')
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size != 2 or not np.isfinite(values).all():
        raise RuntimeError('Invalid inference output')
    return {'class_id': int(values.argmax()),
            'engine_forward_ms': round((time.perf_counter() - t1) * 1000, 4),
            'engine_pipeline_ms': round((time.perf_counter() - t0) * 1000, 4)}


@app.post('/infer')
def infer(request: Request, body: InferRequest):
    local_only(request)
    if ENGINE is None or META is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    started = time.perf_counter()
    try:
        result = inference(ENGINE, body.path, body.text)
    except Exception as exc:
        # Test service intentionally reports failure, never silently switches models.
        raise HTTPException(status_code=503, detail='Inference failed: ' + type(exc).__name__) from exc
    return {'path': body.path, 'class_id': result['class_id'],
            'label': ('NEGATIVE', 'POSITIVE')[result['class_id']],
            'engine_forward_ms': result['engine_forward_ms'],
            'engine_pipeline_ms': result['engine_pipeline_ms'],
            'server_processing_ms': round((time.perf_counter() - started) * 1000, 4),
            'run_id': META['run_id'], 'production_deployed': False}
