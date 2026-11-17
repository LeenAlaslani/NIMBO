"""Local M11 API; preserves M7 reads for historical jobs and its existing execution."""
from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from app.nimbo_agentic_core_v1 import available_agent_config, AgentConfigurationError, probe_local_model
from app import nimbo_agentic_jobs_v1 as jobs
from app.nimbo_resource_monitor_v1 import sample_job_resources
from app import nimbo_auto_pipeline_jobs_v1 as legacy

router=APIRouter(prefix='/nimbo/agentic',tags=['agentic-local-development'])

class AgenticRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model_id:str=Field(min_length=3,max_length=193)
    revision:str=Field(pattern=r'^[0-9a-f]{40}$')
    pilot_file:str=Field(min_length=5,max_length=96)
    heldout_file:str=Field(min_length=5,max_length=96)
    preprocess:str='identity'
    objective:str='speed'

def _root(request:Request)->Path:
    if request.client is None or request.client.host not in ('127.0.0.1','::1','localhost'):
        raise HTTPException(403,'Agentic optimizer is loopback-only')
    return Path.cwd().resolve()

@router.get('/ready')
def ready(request:Request):
    _root(request)
    try:
        settings=available_agent_config()
        return {'ready':probe_local_model(settings),'mode':'multi_agent_llm','model':settings['model'],
                'limits':'Only M7 speed/ONNX FP32 tool is connected. Local reasoning model must be running.'}
    except AgentConfigurationError as e:
        return {'ready':False,'mode':'legacy_only','reason':str(e)}

@router.post('/jobs',status_code=202)
def start(request:Request, body:AgenticRequest):
    root=_root(request)
    try:
        settings=available_agent_config()
        if not probe_local_model(settings):raise AgentConfigurationError('Local reasoning model is offline or missing; no agentic work was started')
        return jobs.submit(root,body.model_dump(),model_config=settings)
    except AgentConfigurationError as e:raise HTTPException(503,str(e)) from e
    except RuntimeError as e:raise HTTPException(409,str(e)) from e
    except (ValueError,OSError) as e:raise HTTPException(422,str(e)) from e

def _dispatch(root,job_id,fn,legacy_fn,*args):
    try:return fn(root,job_id,*args)
    except FileNotFoundError:
        try:return legacy_fn(root,job_id,*args)
        except (FileNotFoundError,ValueError):raise HTTPException(404,'Unknown job')
    except ValueError:raise HTTPException(404,'Invalid or unavailable job')

@router.get('/jobs/{job_id}')
def status(request:Request,job_id:str):
    return _dispatch(_root(request),job_id,jobs.read_job,legacy.read_job)

@router.get('/jobs/{job_id}/events')
def events(request:Request,job_id:str,after:int=Query(default=0,ge=0)):
    return _dispatch(_root(request),job_id,jobs.read_events,legacy.read_events,after)

@router.get('/jobs/{job_id}/resources')
def resources(request: Request, job_id: str):
    # Live local worker-process telemetry, never model benchmarks or replay claims.
    try:
        return sample_job_resources(_root(request), job_id, 'agentic')
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, 'Unknown job or unavailable resource record')

@router.get('/jobs/{job_id}/package')
def package(request:Request,job_id:str):
    try:
        output=_dispatch(_root(request),job_id,jobs.package_for_download,legacy.package_for_download)
    except HTTPException:raise
    except (ValueError,OSError,KeyError):raise HTTPException(404,'No checksum-verified review package')
    return FileResponse(str(output),filename=output.name,media_type='application/zip')
