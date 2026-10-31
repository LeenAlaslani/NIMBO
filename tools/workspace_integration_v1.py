"""Read frontend + real NIMBO API contract; run one pair and verify Evidence Center receipt.

Non-browser integration proof. No benchmark/accuracy or production-throughput claim.
Only localhost endpoints are allowed. Does not alter policy, registry, or adoption state.
"""
from __future__ import annotations
import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

BASELINE = 'textattack/bert-base-uncased-SST-2'
CANDIDATE = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
TEXTS = ['The acting was wonderful and the movie was genuinely moving.',
         'The story was dull and the ending was disappointing.']
OUT = Path('results/workspace_integration_v1')


def loopback_url(s: str) -> str:
    p = urlparse(s)
    if p.scheme != 'http' or p.hostname not in ('localhost','127.0.0.1') or not p.port or p.username or p.password or p.path not in ('', '/') or p.query or p.fragment:
        raise ValueError('Only local HTTP host:port endpoints are allowed, without paths or credentials')
    return s.rstrip('/')


def request_json(url: str, *, method='GET', payload=None, timeout=15):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    req=Request(url,data=data,method=method,headers={'Content-Type':'application/json','Cache-Control':'no-store'})
    with urlopen(req,timeout=timeout) as r:
        raw=r.read(8_000_000)
        if r.status!=200:raise RuntimeError(f'{url}: HTTP {r.status}')
    return json.loads(raw)


def request_text(url:str, *, timeout=15):
    with urlopen(Request(url,headers={'Cache-Control':'no-store'}),timeout=timeout) as r:
        raw=r.read(5_000_000)
        if r.status!=200:raise RuntimeError(f'{url}: HTTP {r.status}')
    return raw.decode('utf-8')


def parse_stream(text: str) -> dict:
    packets=[]
    for i,line in enumerate(text.splitlines(),1):
        if not line.strip():continue
        try:packet=json.loads(line)
        except json.JSONDecodeError as exc:raise ValueError(f'Broken NDJSON at line {i}') from exc
        if not isinstance(packet,dict) or packet.get('type') not in ('event','result','error'):
            raise ValueError(f'Unexpected packet at line {i}')
        packets.append(packet)
    errors=[p for p in packets if p['type']=='error']
    if errors:raise ValueError('Backend emitted error: '+str(errors[0].get('message')))
    results=[p['report'] for p in packets if p['type']=='result']
    if len(results)!=1:raise ValueError(f'Expected exactly one result, got {len(results)}')
    stages=[p.get('stage') for p in packets if p['type']=='event']
    needed=('analyze','route','execute','verify','deliver')
    if any(name not in stages for name in needed):raise ValueError('Missing real events: '+str([n for n in needed if n not in stages]))
    report=results[0]
    if not isinstance(report,dict):raise ValueError('Malformed report')
    comp=report.get('comparison',{})
    if comp.get('sample_count')!=2 or comp.get('all_labels_present') is not False:
        raise ValueError('Unexpected comparison count or label status')
    if comp.get('baseline_model')!=BASELINE or comp.get('candidate_model')!=CANDIDATE:
        raise ValueError('Baseline/candidate mismatch')
    if report.get('guardrail',{}).get('action')!='UNVERIFIED':
        raise ValueError('Unlabeled pair was incorrectly represented as verified')
    if comp.get('evidence_recorded') is not True:
        raise ValueError('Backend did not acknowledge evidence persistence')
    if not isinstance(comp.get('baseline',{}).get('mean_inference_ms'),(int,float)) or not isinstance(comp.get('candidate',{}).get('mean_inference_ms'),(int,float)):
        raise ValueError('Missing measured inference latency')
    return {'event_stages':stages,'reported_latency_ms':{'baseline':comp['baseline']['mean_inference_ms'],'candidate':comp['candidate']['mean_inference_ms']},'guardrail':report['guardrail']['action'],'evidence_recorded':True}


def find_receipt(data: dict, *, timestamp: str, expected_count: int=2) -> bool:
    if not isinstance(data,dict) or not isinstance(data.get('records'),list) or not isinstance(data.get('runtime_summary'),dict):
        raise ValueError('Evidence Center schema mismatch')
    return any(isinstance(r,dict) and r.get('source')=='baseline_comparisons_v15.jsonl' and r.get('timestamp_utc')==timestamp and r.get('sample_count')==expected_count and r.get('all_labels_present') is False and r.get('quality_certified') is False for r in data['records'])


def run(frontend: str, backend: str) -> dict:
    frontend=loopback_url(frontend);backend=loopback_url(backend)
    health=request_json(backend+'/api/health')
    if health.get('ok') is not True:raise RuntimeError('Backend health check did not report ok')
    if health.get('policy_available') is not True:raise RuntimeError('Active classification policy is missing')
    catalog=request_json(backend+'/nimbo/bench/catalog')
    known={m.get('id') for m in catalog.get('models',[]) if isinstance(m,dict)}
    if {BASELINE,CANDIDATE}-known:raise RuntimeError('The UI comparison models are not available in backend catalog')
    validate=request_json(backend+'/nimbo/validate/v21/catalog')
    if validate.get('phase')!='heldout':raise RuntimeError('Heldout validation catalog unreachable/inconsistent')
    page=request_text(frontend+'/')
    if 'id="root"' not in page:raise RuntimeError('Vite returned a page without React root')
    # Vite serves transformed source code on the active development server.
    mission=request_text(frontend+'/src/GameMissionFloor.jsx')
    evidence_js=request_text(frontend+'/src/EvidenceDashboard.jsx')
    workspace=request_text(frontend+'/src/CompanyWorkspace.jsx')
    if '/nimbo/bench/compare' not in mission or '/nimbo/evidence/unified' not in evidence_js or 'Enter Mission Floor' not in workspace:
        raise RuntimeError('Frontend currently served by Vite is missing a NIMBO integration route')
    before=request_json(backend+'/nimbo/evidence/unified?limit=500')
    if not isinstance(before.get('records'),list):raise RuntimeError('Evidence endpoint offline before mission')
    payload={'baseline_model':BASELINE,'candidate_model':CANDIDATE,
             'examples':[{'text':x,'expected_label':None} for x in TEXTS]}
    # This is the exact endpoint and payload contract used by the Workspace compare path.
    stream=request_text_post(backend+'/nimbo/bench/compare',payload)
    parsed=parse_stream(stream)
    latest=request_json(backend+'/nimbo/bench/recent?limit=1')
    entries=latest.get('records',[])
    if not entries or not isinstance(entries[-1],dict):raise RuntimeError('Comparison recent log not updated')
    rec=entries[-1]
    if rec.get('baseline_model')!=BASELINE or rec.get('candidate_model')!=CANDIDATE or rec.get('sample_count')!=2 or rec.get('quality_labels_supplied') is not False:
        raise RuntimeError('Recent evidence does not match submitted unlabeled pair')
    timestamp=rec.get('timestamp_utc')
    if not isinstance(timestamp,str) or not timestamp:raise RuntimeError('Missing evidence timestamp')
    after=request_json(backend+'/nimbo/evidence/unified?limit=500')
    if not find_receipt(after,timestamp=timestamp):raise RuntimeError('Evidence Center did not expose the recorded comparison')
    return {'schema_version':'1.0','status':'PASS','proof_type':'live_frontend_http_contract_plus_backend_inference_and_evidence','browser_clicks_automated':False,
            'run_id':'workspace_'+uuid.uuid4().hex[:12], 'checked_utc':datetime.now(timezone.utc).isoformat(),
            'backend_health':True,'frontend_vite_responding':True,'ui_route_contract':True,
            'comparison':parsed,'evidence_center_receipt_visible':True,'evidence_timestamp_utc':timestamp,
            'source_models':{'baseline':BASELINE,'candidate':CANDIDATE},
            'limitations':['Checks served Vite source and HTTP endpoints, not actual mouse clicks/rendering of 3D robots',
                           'No accuracy claim: this two-input live smoke intentionally omits labels',
                           'No compute savings claim from this two-input smoke',
                           'Appends exactly one aggregate metadata record to the running backend comparison log',
                           'Not a fresh held-out model validation, CPU benchmarking or image quality test']}


def request_text_post(url: str, payload:dict) -> str:
    data=json.dumps(payload).encode('utf8')
    req=Request(url,data=data,method='POST',headers={'Content-Type':'application/json','Cache-Control':'no-store'})
    with urlopen(req,timeout=240) as r:
        if r.status!=200:raise RuntimeError(f'Backend HTTP {r.status}')
        return r.read(8_000_000).decode('utf8')


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description='Live local NIMBO UI/API/Evidence integration; no browser automation')
    p.add_argument('--backend',default='http://127.0.0.1:8000')
    p.add_argument('--frontend',default='http://127.0.0.1:5173')
    a=p.parse_args(argv)
    try:
        result=run(a.frontend,a.backend)
        OUT.mkdir(parents=True,exist_ok=True)
        path=OUT/('workspace_'+result['run_id']+'.json')
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        print('=== NIMBO LIVE WORKSPACE HTTP CONTRACT PROOF ===')
        print('STATUS: PASS')
        print('FRONTEND / BACKEND / MODEL INFERENCE / EVIDENCE: PASS')
        print('STREAM EVENTS:',', '.join(result['comparison']['event_stages']))
        print('GUARDRAIL:',result['comparison']['guardrail'],'(correct: no labels)')
        print('EVIDENCE CENTER RECEIPT: PASS')
        print('BROWSER CLICK TEST: MANUAL STEP STILL REQUIRED')
        print('RECORD:',path)
        return 0
    except (ValueError,RuntimeError,HTTPError,URLError,TimeoutError,OSError,UnicodeError,json.JSONDecodeError) as exc:
        print('=== NIMBO LIVE WORKSPACE HTTP CONTRACT PROOF ===',file=sys.stderr)
        print('STATUS: FAIL',file=sys.stderr)
        print('CAUSE:',f'{type(exc).__name__}: {exc}',file=sys.stderr)
        print('Do not interpret this run as successful; see docs/WORKSPACE_INTEGRATION_STEP20.md',file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
