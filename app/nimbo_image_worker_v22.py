"""NIMBO Image v22: real local Stable Diffusion 1.5 + LCM-LoRA GPU worker.

Run with: python -m uvicorn app.nimbo_image_worker_v22:app --host 127.0.0.1 --port 8011
Public HF assets are downloaded only after the first /generate call. No remote user-supplied URLs.
"""
from __future__ import annotations
import base64
import io
import os
import threading
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
LORA = 'latent-consistency/lcm-lora-sdv1-5'
app = FastAPI(title='NIMBO Local Image GPU Worker', version='22.0')
INFERENCE_LOCK = threading.Lock()

class Generation(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    mode: str = 'paired'
    seed: int = Field(default=42, ge=0, le=2**31-1)

def device_info():
    try:
        import torch
        if torch.cuda.is_available(): return 'cuda', ''
        if torch.backends.mps.is_available(): return 'mps', ''
        if os.getenv('NIMBO_ALLOW_CPU_IMAGE') == '1': return 'cpu','CPU is explicitly enabled; expected to be slow'
        return None, 'No CUDA or MPS GPU detected; GPU worker will not simulate inference'
    except ImportError:
        return None, 'PyTorch unavailable in the active Python environment'

@app.get('/health')
def health():
    dev,reason=device_info()
    return {'ok': bool(dev), 'device':dev, 'reason':reason, 'model':MODEL,
            'lora':LORA, 'models_loaded': engine.cache_info().currsize>0,
            'paired_baseline_available': bool(dev), 'quality_certified':False}

@lru_cache(maxsize=1)
def engine():
    # Heavy imports and downloads happen ONLY at actual inference, not at app startup.
    import torch
    from diffusers import StableDiffusionPipeline, LCMScheduler, DPMSolverMultistepScheduler
    dev,reason=device_info()
    if not dev: raise RuntimeError(reason)
    dtype=torch.float32 if dev=='cpu' else torch.float16
    pipeline=StableDiffusionPipeline.from_pretrained(MODEL,torch_dtype=dtype, use_safetensors=True)
    pipeline.load_lora_weights(LORA,adapter_name='lcm')
    pipeline=pipeline.to(dev)
    pipeline.set_progress_bar_config(disable=True)
    # For 16-GB Macs, reduce intermediate attention memory instead of loading two pipelines.
    if dev=='mps': pipeline.enable_attention_slicing()
    original_scheduler=pipeline.scheduler
    return pipeline,original_scheduler,dev,LCMScheduler,DPMSolverMultistepScheduler

def _sync(torch,dev):
    if dev=='mps': torch.mps.synchronize()
    elif dev=='cuda': torch.cuda.synchronize()

def _image_png(image):
    f=io.BytesIO();image.save(f,format='PNG');payload=f.getvalue()
    if len(payload)>8_000_000: raise RuntimeError('Generated PNG exceeds 8MB')
    return base64.b64encode(payload).decode('ascii')

def infer_one(prompt,mode,seed):
    import torch
    pipe,original,dev,LCMScheduler,DPMSolver=engine()
    if mode=='fast':
        pipe.enable_lora()
        pipe.set_adapters(['lcm'],adapter_weights=[1.0])
        pipe.scheduler=LCMScheduler.from_config(original.config)
        steps,scale=4,1.0
    elif mode=='standard':
        pipe.disable_lora()
        pipe.scheduler=DPMSolver.from_config(original.config)
        steps,scale=25,7.5
    else: raise ValueError('Unexpected image mode')
    # A CPU-based generator is compatible with CUDA and MPS, so both runs use the same seed.
    gen=torch.Generator(device='cpu').manual_seed(seed)
    _sync(torch,dev);start=time.perf_counter()
    with torch.inference_mode():
        output=pipe(prompt=prompt, width=512, height=512,
            num_inference_steps=steps, guidance_scale=scale, generator=gen)
    _sync(torch,dev)
    inference_ms=(time.perf_counter()-start)*1000
    if not getattr(output,'images',None): raise RuntimeError('Diffusion pipeline did not return an image')
    return {'image_base64':_image_png(output.images[0]), 'inference_ms':round(inference_ms,3),
            'steps':steps, 'guidance_scale':scale, 'device':dev}

@app.post('/generate')
def generate(request:Generation):
    if request.mode not in ('fast','standard','paired'):
        raise HTTPException(422,'Allowed modes: fast, standard, paired')
    dev,reason=device_info()
    if not dev:raise HTTPException(503,reason)
    if not INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(409,'GPU is busy with another image. Retry after completion.')
    try:
        started=time.perf_counter()
        results={};trace=[]
        sequence=['fast','standard'] if request.mode=='paired' else [request.mode]
        for stage in sequence:
            trace.append({'stage':'execute','message':f'Generating {stage} image with real {dev.upper()} diffusion','at_ms':round((time.perf_counter()-started)*1000,2)})
            results[stage]=infer_one(request.prompt,stage,request.seed)
        trace.append({'stage':'verify','message':'Both images generated; critical-object quality not automatically certified' if request.mode=='paired' else 'Image generated; human quality not yet evaluated','at_ms':round((time.perf_counter()-started)*1000,2)})
        return {'mode':request.mode,'model':MODEL,'lora':LORA,'seed':request.seed,
                'width':512,'height':512,'results':results,
                'total_worker_ms':round((time.perf_counter()-started)*1000,2),
                'trace':trace,'quality_verified':False}
    except Exception as exc:
        # Keep model stack traces out of public HTTP errors. Worker console holds full details.
        import traceback;traceback.print_exc()
        raise HTTPException(500,f'Image inference failed ({type(exc).__name__}); see GPU worker terminal') from exc
    finally: INFERENCE_LOCK.release()
