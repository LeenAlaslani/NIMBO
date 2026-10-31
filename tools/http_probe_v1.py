"""Bounded, repeatable localhost HTTP load probe against isolated verified engines.

Starts a temporary Uvicorn server on an unused loopback port. Makes actual HTTP
requests, then stops the server. Does NOT touch the running NIMBO API or canary.
All latency and throughput results are descriptive, not approval gates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

from tools.concurrency_probe_v1 import make_prompts, percentile_ms

ROOT = Path('results/http_probe_v1')
LEVELS = (1, 2, 4)


def choose_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def request_json(url, payload=None, timeout=10.0):
    data = json.dumps(payload).encode('utf8') if payload is not None else None
    headers = {'Content-Type': 'application/json', 'User-Agent': 'NIMBO-local-benchmark/1.0'}
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method='POST' if data is not None else 'GET')
    with urllib.request.urlopen(req, timeout=timeout) as res:
        value = res.read(10240)
        if res.status != 200:
            raise RuntimeError(f'HTTP {res.status}')
    return json.loads(value)


def await_server(proc, base, log_path, *, timeout=120):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f'HTTP harness exited during startup ({proc.returncode}); see {log_path}')
        try:
            status = request_json(base + '/health', timeout=1)
            if status.get('ready') is True and status.get('service_kind') == 'isolated_http_probe':
                return status
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass
        time.sleep(0.3)
    raise TimeoutError(f'HTTP harness startup timed out; see {log_path}')


def summarize(rows, elapsed_ms, path, concurrency):
    errors = [r for r in rows if r['error']]
    passed = [r for r in rows if not r['error']]
    result = {
        'path':path, 'concurrency':concurrency, 'requests':len(rows),
        'successes':len(passed), 'errors':len(errors),
        'throughput_successes_per_sec':round(len(passed)*1000/elapsed_ms,4),
        'burst_response_p95_ms':round(percentile_ms([r['elapsed_ms'] for r in passed]),4) if passed else None,
        'server_processing_p95_ms':round(percentile_ms([r['server_ms'] for r in passed]),4) if passed else None,
        'batch_elapsed_ms':round(elapsed_ms,4),
        'error_details':[{'index':r['index'],'error':r['error']} for r in errors[:5]],
    }
    return result, {r['index']:r['prediction'] for r in passed}


def run_http_batch(base, path, prompts, workers, *, request_fn=request_json):
    """Submit matched requests to localhost; clocks include queue and HTTP overhead."""
    if path not in ('baseline','candidate') or workers not in LEVELS or not 8<=len(prompts)<=64:
        raise ValueError('Invalid HTTP batch specification')
    started = time.perf_counter()
    rows = []
    def work(index, text, submitted):
        try:
            resp = request_fn(base+'/infer',{'path':path,'text':text},timeout=20)
            pred = resp['class_id']
            ms = resp['server_processing_ms']
            if (resp.get('path')!=path or type(pred)!=int or pred not in (0,1)
                    or not isinstance(ms,(int,float)) or not math.isfinite(ms) or ms<0
                    or resp.get('production_deployed') is not False):
                raise ValueError('Malformed HTTP result')
            return {'index':index,'elapsed_ms':(time.perf_counter()-submitted)*1000,
                    'server_ms':float(ms),'prediction':pred,'error':None}
        except Exception as exc:
            return {'index':index,'elapsed_ms':(time.perf_counter()-submitted)*1000,
                    'server_ms':0.,'prediction':None,
                    'error':type(exc).__name__+': '+str(exc)[:160]}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fs = {}
        for i, text in enumerate(prompts):
            submitted=time.perf_counter()
            fs[pool.submit(work,i,text,submitted)] = i
        for future in as_completed(fs):
            rows.append(future.result())
    elapsed_ms=(time.perf_counter()-started)*1000
    return summarize(rows,elapsed_ms,path,workers)


def evaluate_condition(base, prompts, workers, repeat):
    order = ('baseline','candidate') if repeat%2==0 else ('candidate','baseline')
    stats={}; labels={}
    for path in order:
        stats[path], labels[path] = run_http_batch(base,path,prompts,workers)
    errors = stats['baseline']['errors'] + stats['candidate']['errors']
    missing = sorted(set(range(len(prompts))) - set(labels['baseline']) - set(labels['candidate']))
    disagreement = [i for i in range(len(prompts)) if labels['baseline'].get(i) != labels['candidate'].get(i)]
    parity = not missing and not disagreement and not errors and len(labels['baseline'])==len(prompts) and len(labels['candidate'])==len(prompts)
    flags=[]
    if parity:
        if stats['candidate']['throughput_successes_per_sec']<stats['baseline']['throughput_successes_per_sec']:
            flags.append('candidate_throughput_worse')
        if stats['candidate']['burst_response_p95_ms']>stats['baseline']['burst_response_p95_ms']:
            flags.append('candidate_burst_p95_worse')
    return {'concurrency':workers,'repeat':repeat+1,'baseline':stats['baseline'],
            'candidate':stats['candidate'], 'predictions_match':bool(parity),
            'mismatch_indices':disagreement[:10], 'performance_flags':flags,
            'status':'FAIL' if not parity else 'PERFORMANCE_REVIEW' if flags else 'PASS'}


def run_probe(requests=24,repeats=2):
    if not 8<=requests<=64 or not 1<=repeats<=3:raise ValueError('Requests 8..64, repeats 1..3')
    prompts=make_prompts(requests)
    ROOT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    log_path=ROOT/f'server_{stamp}.log'
    port=choose_local_port()
    base=f'http://127.0.0.1:{port}'
    # Inherited Python interpreter is the user's existing NIMBO venv.
    env=os.environ.copy()
    env['HF_HUB_OFFLINE']='1'
    env['TRANSFORMERS_OFFLINE']='1'
    with log_path.open('w',encoding='utf8') as log:
        proc=subprocess.Popen([sys.executable,'-m','uvicorn','tools.http_probe_service_v1:app',
            '--host','127.0.0.1','--port',str(port),'--workers','1','--no-access-log'],
            cwd=Path.cwd(),env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            status=await_server(proc,base,log_path)
            results=[]
            for workers in LEVELS:
                for repeat in range(repeats):
                    row=evaluate_condition(base,prompts,workers,repeat)
                    results.append(row)
                    ba=row['baseline'];ca=row['candidate']
                    print(f"HTTP workers {workers}, repeat {repeat+1}: {row['status']} | "
                          f"throughput PyTorch {ba['throughput_successes_per_sec']:.2f}/s, ONNX {ca['throughput_successes_per_sec']:.2f}/s | "
                          f"p95 PyTorch {ba['burst_response_p95_ms']}, ONNX {ca['burst_response_p95_ms']} ms",flush=True)
            fail=sum(r['status']=='FAIL' for r in results)
            flags=sum(bool(r['performance_flags']) for r in results)
            final='FAIL' if fail else 'PERFORMANCE_REVIEW' if flags else 'PASS'
            report={
                'schema_version':'local_http_stress_v1',
                'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                'status':final, 'run_id':status['run_id'],
                'model_revision':status['model_revision'], 'artifact_sha256':status['artifact_sha256'],
                'baseline_threads':status['baseline_threads'], 'candidate_threads':status['candidate_threads'],
                'requests_per_condition':requests,'repeats_per_concurrency':repeats,
                'levels':list(LEVELS),'cases':results,
                'failed_conditions':fail,'flagged_conditions':flags,
                'synthetic_input_sha256':hashlib.sha256('\n'.join(prompts).encode('utf8')).hexdigest(),
                'production_deployed':False,'adoption_state_changed':False,
                'measurement_note':('Real loopback HTTP requests to a temporary, read-only proof FastAPI service wrapping '
                    'the verified staged engines. NOT the primary NIMBO mission_api or production adoption route. '
                    'Client response latency includes HTTP+queue+inference; server processing excludes socket transport. '
                    'Fixed small synthetic unlabeled workload; no labeled accuracy, energy, memory-per-path, cloud cost, '
                    'cold-start, or production-capacity claims. Performance flags descriptive only, NOT statistical gates.'),
            }
            out=ROOT/f'http_stress_{status["run_id"]}_{stamp}.json'
            out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
            print('=== NIMBO LOCAL HTTP PROBE ===')
            print('STATUS:',final)
            print('HTTP SERVICE: Temporary loopback service, stopped after test')
            print('FAILED CONDITIONS:',fail,'PERFORMANCE FLAGS:',flags)
            print('RUN:',status['run_id'])
            print('EVIDENCE:',out)
            if fail:raise RuntimeError('HTTP probe detected request failures or prediction disagreement')
            return report
        finally:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait(timeout=5)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--requests',type=int,default=24)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args(argv)
    run_probe(args.requests,args.repeats)


if __name__=='__main__':
    main()
