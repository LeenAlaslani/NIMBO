"""Local-only review API for M7a real auto-optimizer jobs.

This router is intentionally not a public multi-tenant deployment service.
"""
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from app.nimbo_auto_pipeline_jobs_v1 import submit, read_job, read_events, package_for_download

router = APIRouter(prefix='/nimbo/auto', tags=['auto-optimizer-local'])

class LocalJobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_id: str = Field(min_length=3, max_length=193)
    revision: str = Field(pattern=r'^[0-9a-f]{40}$')
    pilot_file: str = Field(min_length=5, max_length=96)
    heldout_file: str = Field(min_length=5, max_length=96)
    preprocess: str = 'identity'
    objective: str = 'speed'


def _local(request: Request) -> Path:
    if request.client is None or request.client.host not in ('127.0.0.1', '::1', 'localhost'):
        raise HTTPException(403, 'M7 is local-only; remote clients are not accepted')
    return Path.cwd().resolve()

@router.post('/jobs', status_code=202)
def create_auto_job(request: Request, payload: LocalJobRequest):
    root = _local(request)
    try:
        return submit(root, payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc

@router.get('/jobs/{job_id}')
def get_auto_job(request: Request, job_id: str):
    root = _local(request)
    try:
        return read_job(root,job_id)
    except (FileNotFoundError,ValueError) as exc:
        raise HTTPException(404,'Unknown job ID') from exc

@router.get('/jobs/{job_id}/events')
def get_auto_job_events(request: Request, job_id: str, after: int = Query(default=0,ge=0)):
    root = _local(request)
    try:
        return read_events(root,job_id,after)
    except (FileNotFoundError,ValueError) as exc:
        raise HTTPException(404,'Unknown job ID') from exc

@router.get('/jobs/{job_id}/package')
def get_auto_job_package(request: Request, job_id: str):
    root = _local(request)
    try:
        path = package_for_download(root,job_id)
    except (FileNotFoundError,ValueError,KeyError) as exc:
        raise HTTPException(404,'No checksum-verified package for this job') from exc
    return FileResponse(str(path),filename=path.name,media_type='application/zip')
