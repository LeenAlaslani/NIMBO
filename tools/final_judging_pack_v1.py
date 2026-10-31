#!/usr/bin/env python3
"""Build a public, scope-limited NIMBO judging summary from *local measured evidence*.

Read-only input, fail-closed provenance checks, explicitly whitelisted output fields.
No model download, no benchmark execution, no deployment, no raw logs published.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

LOG = Path('results/independent_validation_v21.jsonl')
DEV = Path('results/resource_tuning_v3/latest_development_selection.json')
STRESS = Path('results/concurrency_probe_v1')
HTTP = Path('results/http_probe_v1')
ADOPTION = Path('results/adoption_proof_v1')
OUTPUT = Path('docs/FINAL_JUDGING_EVIDENCE.json')
MARKDOWN = Path('docs/FINAL_JUDGING_EVIDENCE.md')
BEGIN = '<!-- NIMBO_FINAL_PROOF_BEGIN -->'
END = '<!-- NIMBO_FINAL_PROOF_END -->'


def check(test, message):
    if not test:
        raise ValueError('Evidence rejected: ' + message)


def read_json(path):
    check(path.is_file() and not path.is_symlink(), f'missing/unsafe source {path}')
    data=json.loads(path.read_text(encoding='utf8'))
    check(isinstance(data, dict), f'invalid object {path}')
    return data


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def num(value, name, minimum=None, maximum=None):
    check(type(value) in (int,float) and math.isfinite(value), 'invalid '+name)
    if minimum is not None:check(value>=minimum,'negative/out of range '+name)
    if maximum is not None:check(value<=maximum,'out of range '+name)
    return float(value)


def fmt(value):
    return f'{100*value:.2f}%'


def nearest(root, pattern, predicate):
    check(root.is_dir() and not root.is_symlink(), f'missing source directory {root}')
    found=[]
    for file in root.glob(pattern):
        if file.is_symlink() or not file.is_file():continue
        data=read_json(file)
        if predicate(data):found.append((file,data))
    check(bool(found), f'no matching evidence in {root}')
    # Local files are timestamped, and modification time reflects actual creation on this Mac.
    return max(found,key=lambda item:(item[0].stat().st_mtime_ns,item[0].name))


def read_evidence(root=Path('.')):
    root=Path(root)
    def rp(p):return root/p
    dev_path=rp(DEV)
    dev=read_json(dev_path)
    check(dev.get('status')=='SELECTED_DEV_ONLY' and dev.get('heldout_evaluated_for_this_selection') is True,
          'development selection must be locked and heldout-evaluated')
    chosen=dev.get('selected_candidate_threads')
    check(type(chosen) is int and 1<=chosen<=8,'no valid development-selected threads')
    results=[r for r in dev.get('results',[]) if r.get('candidate_threads')==chosen]
    check(len(results)==1 and results[0].get('gate',{}).get('eligible_on_development') is True,
          'selected candidate did not pass development gate')
    check(dev.get('development_samples')==200 and dev.get('development_rounds')==2,
          'development sample count/rounds unverified')
    log=rp(LOG)
    check(log.is_file(),'heldout results log missing')
    lines=[json.loads(s) for s in log.read_text(encoding='utf8').splitlines() if s.strip()]
    check(bool(lines),'heldout results log empty')
    # Don't cherry-pick earlier successful runs: newest validation run must match the locked selection.
    trial=lines[-1]
    run=trial.get('run_id')
    check(isinstance(run,str) and run.startswith('heldout_'),'invalid heldout ID')
    check(trial.get('sample_count')==800 and trial.get('rounds')==3 and
          trial.get('candidate_thread_count')==chosen and trial.get('thread_count')==4,
          'latest heldout result does not use locked 4-vs-selected-thread setup')
    check(trial.get('model_id')==dev.get('model_id') and
          trial.get('model_revision')==dev.get('model_revision'),
          'model ID/revision mismatch between development and heldout')
    comp=trial.get('comparison',{})
    gate=trial.get('optimization',{}).get('gate',{})
    export=trial.get('optimization',{}).get('export') or {}
    check(trial.get('optimization',{}).get('approval')=='eligible_for_review',
          'heldout candidate not eligible for review')
    check(gate.get('optimization_objective')=='cpu_time' and
          all(gate.get(k) is True for k in ('eligible','quality_pass','latency_pass','cpu_pass','p95_pass','repeatability_pass')),
          'CPU-time objective/quality/latency/repeatability gate did not pass')
    check(comp.get('sample_count')==800 and comp.get('rounds')==3 and
          comp.get('prediction_agreement_fraction')==1,
          'paired sample counts or agreement not verified')
    baseline=comp.get('baseline',{}); candidate=comp.get('candidate',{})
    a=num(baseline.get('accuracy_on_reference_labels'),'baseline accuracy',0,1)
    b=num(candidate.get('accuracy_on_reference_labels'),'candidate accuracy',0,1)
    ncorrect=baseline.get('correct_unique'); ncorrect2=candidate.get('correct_unique')
    check(type(ncorrect) is int and type(ncorrect2) is int and ncorrect==ncorrect2 and
          abs(ncorrect/800-a)<1e-6 and abs(ncorrect2/800-b)<1e-6,'accuracy/correct counts mismatch')
    wall=num(gate.get('mean_speedup_fraction'),'wall reduction',0,1)
    cpu=num(gate.get('cpu_time_reduction_fraction'),'CPU reduction',0,1)
    wall_b=num(baseline.get('warm_forward',{}).get('mean_ms'),'baseline wall',0)
    wall_c=num(candidate.get('warm_forward',{}).get('mean_ms'),'candidate wall',0)
    cpu_b=num(baseline.get('process_cpu_forward',{}).get('mean_ms'),'baseline CPU',0)
    cpu_c=num(candidate.get('process_cpu_forward',{}).get('mean_ms'),'candidate CPU',0)
    check(wall_b>0 and cpu_b>0 and abs((1-wall_c/wall_b)-wall)<0.003 and
          abs((1-cpu_c/cpu_b)-cpu)<0.003,'reduction ratio inconsistent with measured times')
    checksum=export.get('sha256')
    check(isinstance(checksum,str) and len(checksum)==64 and all(ch in '0123456789abcdef' for ch in checksum),
          'exported ONNX candidate checksum missing')
    check(export.get('manifest') and export.get('onnx_model'), 'export manifest/path not recorded')
    model_revision=trial.get('model_revision')
    def goodline(obj,model_sha_key):
        return (obj.get('run_id',obj.get('source_run_id'))==run and
                obj.get('model_revision')==model_revision and
                obj.get(model_sha_key)==checksum and
                obj.get('candidate_threads',obj.get('onnx_threads'))==chosen and
                obj.get('baseline_threads',obj.get('pytorch_threads'))==4 and
                obj.get('production_deployed') is False)
    stress_path,stress=nearest(rp(STRESS),'stress_*.json',lambda x:goodline(x,'model_sha256'))
    check(stress.get('status')=='PASS' and stress.get('adoption_state_changed') is False,
          'engine-level stress proof failed or mutated state')
    tests=stress.get('tests',[])
    check(len(tests)==6 and {t.get('concurrency') for t in tests}=={1,2,4} and
          all(t.get('status')=='PASS' and t.get('prediction_parity',{}).get('all_agree') is True
              and not t.get('performance_flags') for t in tests),
          'engine-level stress conditions not all green')
    http_path,http=nearest(rp(HTTP),'http_stress_*.json',lambda x:goodline(x,'artifact_sha256'))
    cases=http.get('cases',[])
    check(http.get('status')=='PASS' and http.get('failed_conditions')==0 and http.get('flagged_conditions')==0
          and http.get('adoption_state_changed') is False and len(cases)==6 and
          {t.get('concurrency') for t in cases}=={1,2,4} and
          all(t.get('status')=='PASS' and t.get('predictions_match') is True and
              not t.get('performance_flags') for t in cases),
          'HTTP stress conditions not all green')
    adoption_path,adoption=nearest(rp(ADOPTION),'session_*/report.json',
                                  lambda x:goodline(x,'artifact_sha256'))
    flow=adoption.get('shadow_and_canary') or {}
    check(adoption.get('status')=='PASS' and adoption.get('rollback_verified') is True and
          adoption.get('original_adoption_state_modified') is False and
          adoption.get('test_service_kind')=='isolated_actual_adoption_router_proof' and
          flow.get('outcome')=='CANARY_PASS' and
          flow.get('canary_pairs')==5 and flow.get('canary_candidate_only')==2 and
          not adoption.get('errors'), 'adoption proof has missing safety stages')
    def source(path):
        return {'file':str(path.relative_to(root)), 'sha256':sha(path)}
    manifest={'schema_version':'nimbo_final_judging_evidence_v1',
       'source_run_id':run, 'model_id':trial['model_id'],
       'model_revision':model_revision, 'artifact_sha256':checksum,
       'dataset_partition_digest':trial.get('dataset_fingerprint'),
       'candidate_status':'eligible_for_review_only',
       'baseline_cpu_threads':4,'candidate_cpu_threads':chosen,
       'holdout_examples':800,'holdout_rounds':3,
       'correct_per_path':ncorrect,'accuracy_baseline':a,'accuracy_candidate':b,
       'prediction_agreement_fraction':1.0,'mean_warm_forward_latency_reduction_fraction':wall,
       'mean_process_cpu_time_reduction_fraction':cpu,
       'baseline_warm_forward_ms':wall_b,'candidate_warm_forward_ms':wall_c,
       'baseline_process_cpu_ms':cpu_b,'candidate_process_cpu_ms':cpu_c,
       'development':{'samples':200,'rounds':2,'selected_threads':chosen,
                      'selection_status':'SELECTED_DEV_ONLY',
                      'candidate_dev_wall_reduction_fraction':results[0]['gate']['wall_improvement_fraction'],
                      'candidate_dev_cpu_reduction_fraction':results[0]['gate']['cpu_time_improvement_fraction']},
       'concurrency':{'scope':'engine_only','status':stress['status'],
                      'levels':[1,2,4],'cases_passed':6},
       'http':{'scope':'temporary_loopback_probe_not_main_api','status':http['status'],
               'levels':[1,2,4],'cases_passed':6},
       'adoption':{'scope':'isolated_actual_router_http_not_primary_ui',
                   'status':adoption['status'], 'shadow_pairs':flow.get('shadow_count'),
                   'canary_pairs':flow['canary_pairs'],'canary_candidate_only':flow['canary_candidate_only'],
                   'rollback_verified':True,'original_state_untouched':True},
       'sources':{'development':source(dev_path),'heldout_log':source(log),
                  'engine_stress':source(stress_path),'http_stress':source(http_path),
                  'adoption':source(adoption_path)},
       'not_measured':['electrical energy','FLOPs','cloud billing','per-path peak RAM',
                       'end-to-end primary web UI performance','external independent audit'],
       'cautions':['2.54% process CPU time reduction is a single measured run; no confidence interval',
                   'heldout IMDb subset was reused in earlier experiments; not a virgin external dataset',
                   'all stages are local and do not imply production deployment',
                   'stress samples are small, synthetic and unlabeled',
                   'Shadow performs paired inference and does not save compute']}
    return manifest


def md_report(m):
    return '\n'.join([
        '# NIMBO — final judging evidence (local measured results)', '',
        f"**Model:** `{m['model_id']}` | **run:** `{m['source_run_id']}`",
        f"**Artifact SHA-256:** `{m['artifact_sha256']}`",
        f"**Baseline:** PyTorch FP32, {m['baseline_cpu_threads']} threads | **Candidate:** ONNX Runtime FP32, {m['candidate_cpu_threads']} threads", '',
        '## Task Performance and compute', '',
        '| Metric | Result | Scope |', '|---|---:|---|',
        f"| Held-out labeled accuracy | {fmt(m['accuracy_baseline'])} → {fmt(m['accuracy_candidate'])} | {m['holdout_examples']} IMDb test-subset texts; {m['holdout_rounds']} paired CPU rounds |",
        f"| Warm forward latency reduction | **{fmt(m['mean_warm_forward_latency_reduction_fraction'])}** | Mean per-text, excludes setup and HTTP |",
        f"| Process CPU-time reduction | **{fmt(m['mean_process_cpu_time_reduction_fraction'])}** | Aggregate user+system CPU time across process threads, measured in one local run |", '',
        f"**Development choice:** {m['development']['samples']} dev examples × {m['development']['rounds']} rounds; selected {m['candidate_cpu_threads']} ONNX threads before checking 800 held-out examples.", '',
        '## Robustness and adoption', '',
        f"- Engine-level CPU concurrency: **{m['concurrency']['cases_passed']}/6 PASS**, workers 1, 2, 4.",
        f"- Temporary localhost HTTP inference: **{m['http']['cases_passed']}/6 PASS**, workers 1, 2, 4. This is not the primary NIMBO UI/API.",
        f"- Isolated real adoption router over HTTP: **{m['adoption']['status']}**; Shadow {m['adoption']['shadow_pairs']} paired requests, Canary {m['adoption']['canary_pairs']} paired safety checks and {m['adoption']['canary_candidate_only']} candidate-only requests, followed by **verified rollback** to PyTorch.",
        '- No customer production deployment occurred. Shadow calls both models and is not compute saving.', '',
        '## Limitations and auditability', '',
        '- The IMDb holdout is disjoint from this tuning set, but was previously evaluated in earlier runs; it is **not** a new independent external test.',
        '- CPU time improved in this measurement, but the small positive margin needs additional runs for statistical confidence; no energy, FLOPs, dollar cost or individual-model RAM savings measured.',
        '- Unlabeled HTTP/concurrency samples establish operation and agreement on that small test, **not** classification accuracy.',
        '- Candidate approval means *eligible for manual review*, not automatically deployed.',
        '- Source-file SHA-256 digests below identify local raw logs; they do not make local logs publicly available. Public summary is a derived evidence index.', '',
        '## Local raw evidence index (hashes only)', '',
        '| Source | Local relative file | SHA-256 |', '|---|---|---|',
        *[f"| {k} | `{v['file']}` | `{v['sha256']}` |" for k,v in m['sources'].items()],
        '', '## How to reproduce', '',
        'See [`CPU_THREAD_TUNING_STEP13.md`](CPU_THREAD_TUNING_STEP13.md), [`HTTP_PROBE_STEP16.md`](HTTP_PROBE_STEP16.md), and [`ADOPTION_PROOF_STEP17.md`](ADOPTION_PROOF_STEP17.md).',
        'Do not rerun the locked held-out tuner merely to improve a number; a genuinely fresh external dataset is a separate next evaluation.', '' ])


def update_readme(text, m):
    summary='\n'.join([BEGIN,'## Primary proof: lower warm inference latency and process CPU time', '',
      f"With PyTorch FP32 (4 threads) versus ONNX Runtime FP32 ({m['candidate_cpu_threads']} threads), the locked local 800-example evaluation measured **{fmt(m['mean_warm_forward_latency_reduction_fraction'])} lower mean warm forward latency** and **{fmt(m['mean_process_cpu_time_reduction_fraction'])} lower process CPU time**, with **{fmt(m['accuracy_candidate'])} accuracy on both paths**. The candidate is *eligible for review*, not deployed. CPU-time savings are small and not an energy or billing measurement.", '',
      'Full lineage and caveats: [`docs/FINAL_JUDGING_EVIDENCE.md`](docs/FINAL_JUDGING_EVIDENCE.md).', END, ''])
    if BEGIN in text or END in text:
        check(text.count(BEGIN)==1 and text.count(END)==1 and text.index(BEGIN)<text.index(END),
              'invalid existing README proof markers')
        return text[:text.index(BEGIN)]+summary.rstrip('\n')+text[text.index(END)+len(END):]
    check('## Verified main proof' in text,'README changed unexpectedly; not overwriting')
    return text.replace('## Verified main proof',summary.rstrip('\n')+'\n\n## Verified main proof',1)


def publish(root=Path('.')):
    root=Path(root).resolve()
    m=read_evidence(root)
    readme=root/'README.md'; check(readme.is_file(),'README missing')
    current=readme.read_text(encoding='utf8')
    updated=update_readme(current,m)
    rendered=md_report(m)
    js=json.dumps(m,indent=2,ensure_ascii=False)+'\n'
    # Keep a local, gitignored README backup, never exported to public docs.
    if current != updated:
        back=root/'results/final_proof_backups'/('README_before_'+hashlib.sha256(current.encode('utf8')).hexdigest()[:12]+'.md')
        back.parent.mkdir(parents=True,exist_ok=True)
        if not back.exists():
            back.write_text(current,encoding='utf8')
    # Atomic writes per destination; all validation completes before touching any output.
    for path,content in [(root/OUTPUT,js),(root/MARKDOWN,rendered),(readme,updated)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+'.tmp')
        temp.write_text(content,encoding='utf8')
        temp.replace(path)
    print('=== NIMBO FINAL JUDGING PROOF PACK ===')
    print('STATUS: PASS - all five local evidence sources verified and linked')
    print('RUN:',m['source_run_id'])
    print('ONNX selected threads:',m['candidate_cpu_threads'])
    print('ACCURACY:',fmt(m['accuracy_baseline']),'->',fmt(m['accuracy_candidate']))
    print('WARM LATENCY REDUCTION:',fmt(m['mean_warm_forward_latency_reduction_fraction']))
    print('PROCESS CPU TIME REDUCTION:',fmt(m['mean_process_cpu_time_reduction_fraction']))
    print('ENGINE CONCURRENCY: 6/6 | LOCAL HTTP: 6/6 | ADOPTION: PASS | ROLLBACK: True')
    print('PUBLIC FILES:',MARKDOWN,OUTPUT,'README.md')
    print('NO GIT COMMIT/PUSH. RAW LOGS REMAIN LOCAL.')
    return m


def main():
    p=argparse.ArgumentParser(description='Validate local proof lineage and write public summary to docs/')
    p.add_argument('--root',type=Path,default=Path('.'))
    a=p.parse_args()
    publish(a.root)

if __name__=='__main__':
    main()
