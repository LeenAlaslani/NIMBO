"""Temporary LOCAL-ONLY proof harness for the actual NIMBO adoption router.

This imports the real adoption endpoints. It replaces ONLY its state/log PATHS
with a session-specific results directory; never rewires production routing.
The proof runner launches it in a separate process bound to loopback.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from app import nimbo_adoption_v21_1 as adoption
from tools.concurrency_probe_v1 import SOURCE_SELECTION, validate_lineage

SERVICE_KIND = 'isolated_actual_adoption_router_proof'
ROOT = (Path.cwd() / 'results' / 'adoption_proof_v1').resolve()


def check_proof_directory(value: str | None) -> Path:
    if not value:
        raise ValueError('NIMBO_ADOPTION_PROOF_DIR must be set by the proof runner')
    candidate = Path(value).resolve()
    if candidate.parent != ROOT or not candidate.name.startswith('session_') or candidate.is_symlink():
        raise ValueError('Refusing adoption state path outside isolated proof directory')
    if not candidate.is_dir():
        raise ValueError('Proof session directory missing')
    return candidate


def set_isolated_paths(directory: Path) -> None:
    adoption.STATE = directory / 'state.json'
    adoption.EVENTS = directory / 'events.jsonl'
    adoption.TRIALS = directory / 'requests.jsonl'


def prepare_proof():
    import json
    from tools.staged_candidate_smoke_v1 import latest_exported_run

    directory = check_proof_directory(os.getenv('NIMBO_ADOPTION_PROOF_DIR'))
    set_isolated_paths(directory)
    if adoption.STATE.exists() or adoption.EVENTS.exists() or adoption.TRIALS.exists():
        raise RuntimeError('Proof session must start with fresh adoption state')
    run_id = latest_exported_run()
    if os.getenv('NIMBO_ADOPTION_PROOF_RUN_ID') != run_id:
        raise RuntimeError('Pinned run ID changed; refusing stale proof')
    review = adoption._review(run_id)  # checksum, model revision, heldout decision
    selection = json.loads(SOURCE_SELECTION.read_text(encoding='utf8'))
    selected_threads = validate_lineage(selection, review['validation'], review)
    engine = adoption._prepare_cache(review)  # actual pinned PyTorch & ONNX CPU engines
    if engine['candidate'].get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('Unexpected candidate runtime provider')
    return {
        'run_id': run_id,
        'model_revision': review['revision'],
        'candidate_sha256': review['sha256'],
        'candidate_threads': selected_threads,
        'baseline_threads': review['validation']['thread_count'],
        'production_deployed': False,
        'changes_existing_nimbo_routing': False,
        'service_kind': SERVICE_KIND,
        'ready': True,
    }


META = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global META
    META = prepare_proof()
    try:
        yield
    finally:
        META = None
        adoption.PROCESS_CACHE.clear()


app = FastAPI(title='NIMBO isolated adoption proof; NOT main mission_api', lifespan=lifespan)
app.include_router(adoption.router)


@app.get('/proof/health')
def health(request: Request):
    if not request.client or request.client.host not in ('127.0.0.1', '::1', 'testclient'):
        raise HTTPException(403, 'Loopback only')
    if request.headers.get('origin'):
        raise HTTPException(403, 'No browser origins accepted')
    if META is None:
        raise HTTPException(503, 'Proof engine not ready')
    return META
