"""NIMBO M5: size-oriented dynamic INT8 candidate from an already reviewed M4 ZIP.

This is a *development-only* trial, NOT an approved downloadable model.
It uses the EXACT FP32 ONNX graph from M4, verifies its manifest and M3 chain,
then performs genuine ONNX Runtime dynamic weight quantization and paired CPU
inference on a separate labeled development dataset.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from app.nimbo_heldout_validation_v1 import load_heldout, preprocess_text
from app.nimbo_model_intake_v1 import resolve_labels, percentile
from app.nimbo_onnx_candidate_v1 import digest, summarize_measurements
from app.nimbo_package_v1 import check_review, verify_bundle_files

MAX_ARCHIVE_BYTES = 950 * 1024 * 1024
MAX_EXTRACT_BYTES = 970 * 1024 * 1024
MAX_MEMBERS = 28


def validate_int8_gate(fp32: dict, int8: dict, *, old_bytes: int, new_bytes: int,
                       sample_count: int, agreement: float, mismatches: int,
                       objective: str = 'size') -> dict:
    """Validate separate size, quality and compute metrics; no release authority."""
    if objective not in {'size', 'balanced'}:
        raise ValueError('Objective must be size or balanced')
    if not 200 <= sample_count <= 800 or old_bytes <= 0 or new_bytes <= 0:
        raise ValueError('Invalid dataset or artifact sizes')
    if not 0 <= agreement <= 1 or mismatches < 0:
        raise ValueError('Invalid consistency measurements')
    for metric in ('accuracy', 'mean_warm_forward_ms','p95_warm_forward_ms','process_cpu_forward_total_ms'):
        a, b = fp32.get(metric), int8.get(metric)
        if type(a) not in (int, float) or type(b) not in (int, float):
            raise ValueError('Missing numeric metric: '+metric)
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Nonfinite metric: '+metric)
        if metric == 'accuracy':
            if not 0 <= a <= 1 or not 0 <= b <= 1:
                raise ValueError('Accuracy must be in [0,1]')
        elif a <= 0 or b <= 0:
            raise ValueError('Nonpositive CPU/latency measurement')
    size_reduction = 1 - new_bytes / old_bytes
    accuracy_drop = fp32['accuracy'] - int8['accuracy']
    mean_change = int8['mean_warm_forward_ms'] / fp32['mean_warm_forward_ms'] - 1
    p95_change = int8['p95_warm_forward_ms'] / fp32['p95_warm_forward_ms'] - 1
    cpu_change = int8['process_cpu_forward_total_ms'] / fp32['process_cpu_forward_total_ms'] - 1
    quality_ok = accuracy_drop <= .01 + 1e-9 and agreement >= .98 - 1e-9 and mismatches == 0
    size_ok = size_reduction >= .20 - 1e-9
    speed_limit = 0.25 if objective == 'size' else -0.05
    p95_limit = 0.25 if objective == 'size' else 0.0
    cpu_limit = 0.25 if objective == 'size' else 0.0
    compute_ok = mean_change <= speed_limit + 1e-9 and p95_change <= p95_limit + 1e-9 and cpu_change <= cpu_limit + 1e-9
    passed = bool(quality_ok and size_ok and compute_ok)
    return {'decision': 'INT8_DEV_PASS_NEEDS_NEW_HELDOUT_AND_RELOAD' if passed else 'INT8_REJECTED_DEV_GATES',
            'dev_gates_passed': passed,'quality_pass':bool(quality_ok),
            'size_pass':bool(size_ok),'performance_pass':bool(compute_ok),
            'graph_size_reduction_fraction':round(size_reduction,6),
            'accuracy_drop':round(accuracy_drop,6),
            'prediction_agreement':round(agreement,6),
            'repeatability_mismatches':mismatches,
            'mean_latency_change_fraction':round(mean_change,6),
            'p95_change_fraction':round(p95_change,6),
            'process_cpu_change_fraction':round(cpu_change,6),
            'limits':{'min_graph_size_reduction':.20,'max_absolute_accuracy_drop':.01,
                      'min_prediction_agreement':.98,'max_mean_latency_increase':speed_limit,
                      'max_p95_increase':p95_limit,'max_cpu_increase':cpu_limit},
            'eligible_for_deployment':False,'eligible_for_downloadable_package':False}


def verify_and_extract_m4(archive_path: Path, target: Path, *, expected_sha256: str,
                          m3: dict, m3_report: Path, fp32_graph_sha256: str) -> tuple[Path, dict]:
    """Fail closed on archive identity, ZIP traversal/bombs, manifest and graph checks."""
    if archive_path.is_symlink() or not archive_path.is_file() or not 0 < archive_path.stat().st_size <= MAX_ARCHIVE_BYTES:
        raise ValueError('M4 archive missing, symlinked, or unreasonably large')
    if len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('Supply the verified lowercase M4 archive SHA-256')
    if digest(archive_path) != expected_sha256:
        raise ValueError('M4 ZIP hash differs from the previously verified archive')
    with zipfile.ZipFile(archive_path) as z:
        members = z.infolist()
        if not members or len(members) > MAX_MEMBERS or any(i.is_dir() for i in members):
            raise ValueError('Unexpected M4 archive structure')
        manifest_name = [i.filename for i in members if i.filename.endswith('/manifest.json')]
        if len(manifest_name) != 1:
            raise ValueError('M4 archive needs exactly one manifest')
        dirname = manifest_name[0][:-len('/manifest.json')]
        if not dirname.startswith('nimbo_onnx_review_') or '/' in dirname or '\\' in dirname:
            raise ValueError('Invalid package directory')
        manifest_info = z.getinfo(manifest_name[0])
        if manifest_info.file_size > 128*1024:
            raise ValueError('Oversized package manifest')
        manifest = json.loads(z.read(manifest_name[0]))
        if manifest.get('schema_version') != 'nimbo_optimized_package_v1' or manifest.get('review_status') != 'LOCAL_REVIEW_ONLY_NOT_DEPLOYED':
            raise ValueError('Not a verified M4 review package')
        files = manifest.get('files')
        if not isinstance(files, dict) or not files or len(files)>24:
            raise ValueError('Invalid M4 manifest files')
        allowed = {dirname+'/'+n for n in files} | {manifest_name[0]}
        names = [i.filename for i in members]
        if len(names) != len(set(names)) or set(names) != allowed:
            raise ValueError('Unexpected ZIP entries or duplicate paths')
        total = 0
        for i in members:
            base = i.filename[len(dirname)+1:]
            if not base or '/' in base or '\\' in base or base in ('.','..'):
                raise ValueError('Unsafe ZIP member name')
            if stat.S_IFMT(i.external_attr >> 16) == stat.S_IFLNK:
                raise ValueError('Symlink ZIP entry refused')
            if i.file_size < 0 or i.file_size > MAX_ARCHIVE_BYTES:
                raise ValueError('Oversized ZIP member')
            total += i.file_size
            if total > MAX_EXTRACT_BYTES:
                raise ValueError('Excessive extracted ZIP size')
        if manifest.get('source_m3_report_sha256') != digest(m3_report) or manifest.get('source_m2_graph_sha256') != fp32_graph_sha256:
            raise ValueError('M4 manifest is not linked to the supplied M3 evidence/graph')
        if manifest.get('heldout_dataset_sha256') != m3.get('heldout_dataset_sha256'):
            raise ValueError('M4/M3 heldout identity mismatch')
        identity = manifest.get('model') or {}
        original = m3.get('model') or {}
        if identity.get('model_id') != original.get('model_id') or identity.get('resolved_revision') != original.get('resolved_revision'):
            raise ValueError('M4/M3 model identity mismatch')
        if manifest.get('preprocessing') != m3.get('preprocessing'):
            raise ValueError('M4/M3 preprocessing mismatch')
        target.mkdir(parents=True, exist_ok=False)
        z.extractall(target)
    folder = target/dirname
    verify_bundle_files(folder,manifest)
    if digest(folder/'model.onnx') != fp32_graph_sha256:
        raise ValueError('FP32 ONNX graph checksum mismatch inside M4 package')
    return folder, manifest


def _session(path: Path, *, threads: int):
    import onnxruntime as ort
    options=ort.SessionOptions()
    options.intra_op_num_threads=threads
    options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('Expected CPUExecutionProvider only')
    inputs=tuple(x.name for x in session.get_inputs())
    if 'input_ids' not in inputs or any(n not in ('input_ids','attention_mask','token_type_ids') for n in inputs):
        raise RuntimeError('Unexpected ONNX input signature')
    if len(session.get_outputs()) != 1:
        raise RuntimeError('Unexpected ONNX output signature')
    return session, inputs


def paired_evaluation(source: Path, quantized: Path, pkg: Path, rows: list[dict],
                      *, manifest: dict, objective: str, rounds: int = 3) -> tuple[dict,dict,dict]:
    import numpy as np
    from transformers import AutoTokenizer
    threads=manifest['cpu_threads']
    length=manifest['max_length']
    labels={int(i):v for i,v in manifest['model']['id2label'].items()}
    expected=resolve_labels(rows,labels)
    if any(e is None for e in expected):
        raise ValueError('All development data must have reference labels')
    tokenizer=AutoTokenizer.from_pretrained(str(pkg),local_files_only=True,trust_remote_code=False)
    baseline, baseline_names=_session(source,threads=threads)
    candidate, candidate_names=_session(quantized,threads=threads)
    if set(baseline_names)!=set(candidate_names):
        raise RuntimeError('Quantized ONNX input signature mismatch')
    prepared=[]
    for row in rows:
        tok=tokenizer(preprocess_text(row['text'],manifest['preprocessing']),max_length=length,
                      truncation=True,return_tensors='np')
        if any(n not in tok for n in baseline_names):
            raise ValueError('Tokenizer does not provide required model inputs')
        prepared.append({name:np.asarray(tok[name],dtype=np.int64) for name in baseline_names})
    for feed in prepared[:3]:
        baseline.run(None,{k:feed[k] for k in baseline_names})
        candidate.run(None,{k:feed[k] for k in candidate_names})
    walls={'fp32':[],'int8':[]}
    cpus={'fp32':[],'int8':[]}
    preds={'fp32':[],'int8':[]}
    repeat={}
    disagreements=0
    mismatches=0
    for r in range(rounds):
        for i, feed in enumerate(prepared):
            pair={}
            for which in (('fp32','int8') if (r+i)%2==0 else ('int8','fp32')):
                session,names=(baseline,baseline_names) if which=='fp32' else (candidate,candidate_names)
                start=time.perf_counter_ns()
                cpu_start=time.process_time_ns()
                result=np.asarray(session.run(None,{k:feed[k] for k in names})[0])
                cpu=(time.process_time_ns()-cpu_start)/1e6
                wall=(time.perf_counter_ns()-start)/1e6
                if result.shape!=(1,len(labels)) or not np.isfinite(result).all():
                    raise RuntimeError('Non-finite or incompatible logits from '+which)
                pred=int(result.argmax(axis=1)[0])
                preds[which].append(pred)
                cpus[which].append(cpu)
                walls[which].append(wall)
                pair[which]=pred
            if pair['fp32'] != pair['int8']:
                disagreements += 1
            current=(pair['fp32'],pair['int8'])
            if i in repeat and repeat[i]!=current:
                mismatches += 1
            repeat[i]=current
    reference=expected*rounds
    metrics_fp32=summarize_measurements(walls['fp32'],cpus['fp32'],reference,preds['fp32'])
    metrics_int8=summarize_measurements(walls['int8'],cpus['int8'],reference,preds['int8'])
    gate=validate_int8_gate(metrics_fp32,metrics_int8,old_bytes=source.stat().st_size,new_bytes=quantized.stat().st_size,
                           sample_count=len(rows),agreement=1-disagreements/(len(rows)*rounds),
                           mismatches=mismatches,objective=objective)
    return metrics_fp32,metrics_int8,gate


def run_int8_trial(archive: str | Path, sha256: str, m3_report: str | Path,
                   m3_dataset: str | Path, development_dataset: str | Path,
                   *, root: str | Path='.', objective: str='size', rounds: int=3,
                   output_root: str | Path='results/auto_optimizer_int8_v1') -> tuple[Path,dict]:
    if not 2 <= rounds <= 5 or objective not in ('size','balanced'):
        raise ValueError('Invalid trial rounds/objective')
    project=Path(root).expanduser().resolve(strict=True)
    output=((project / output_root).resolve() if not Path(output_root).is_absolute() else Path(output_root).expanduser().resolve())
    if output != (project/'results'/'auto_optimizer_int8_v1').resolve():
        raise ValueError('INT8 output must remain under project results directory')
    m3,m2,graph,_,m3_dataset_sha=check_review(m3_report,m3_dataset,root=project)
    m3_path=Path(m3_report).expanduser().resolve(strict=True)
    dev_rows,dev_sha,provenance=load_heldout(development_dataset)
    if dev_sha in (m3_dataset_sha,m2['dataset_sha256']):
        raise ValueError('Development data must differ from M2 and M3 datasets')
    if provenance.get('source')=='cardiffnlp/tweet_eval' and provenance.get('split')!='validation':
        raise ValueError('Use the TweetEval validation split for M5 development, NOT the M3 test split')
    out_dir=output/('int8_'+uuid.uuid4().hex[:16])
    output.mkdir(parents=True,exist_ok=True)
    out_dir.mkdir(parents=False,exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix='.m5_extract_',dir=output) as temp:
            pkg,manifest=verify_and_extract_m4(Path(archive).expanduser().resolve(strict=True),Path(temp)/'extracted',
                expected_sha256=sha256,m3=m3,m3_report=m3_path,fp32_graph_sha256=m2['artifact']['sha256'])
            source=pkg/'model.onnx'
            # ONNX Runtime's dynamic int8 weight quantization actually rewrites MatMul nodes.
            from onnxruntime.quantization import quantize_dynamic,QuantType
            import onnx
            quant_file=out_dir/'model.int8.onnx'
            quantize_dynamic(str(source),str(quant_file),weight_type=QuantType.QInt8,
                             per_channel=False,op_types_to_quantize=['MatMul'])
            if not quant_file.is_file() or quant_file.stat().st_size < 1024:
                raise RuntimeError('Quantization did not produce an ONNX graph')
            graph_proto=onnx.load(str(quant_file),load_external_data=False)
            changed_ops={n.op_type for n in graph_proto.graph.node}
            if not changed_ops.intersection({'MatMulInteger','QLinearMatMul','DynamicQuantizeLinear'}):
                raise RuntimeError('No dynamic INT8 MatMul operators found; refusing no-op conversion')
            fp32,int8,gate=paired_evaluation(source,quant_file,pkg,dev_rows,manifest=manifest,
                                             objective=objective,rounds=rounds)
            result={'schema_version':'nimbo_int8_dev_trial_v1',
                    'status':'INT8_DEVELOPMENT_ONLY_NO_EXPORT',
                    'model':manifest['model'],
                    'source_m4_archive_sha256':sha256,
                    'source_m3_report_sha256':digest(m3_path),
                    'source_fp32_graph_sha256':digest(source),
                    'dev_dataset_sha256':dev_sha,'dev_dataset_provenance':provenance,
                    'dev_unique_examples':len(dev_rows),'rounds':rounds,
                    'objective':objective,'preprocessing':manifest['preprocessing'],
                    'fp32_graph_bytes':source.stat().st_size,
                    'int8_graph_bytes':quant_file.stat().st_size,
                    'int8_graph_sha256':digest(quant_file),
                    'quantization':{'method':'onnxruntime_dynamic','weight_type':'QInt8',
                                    'per_channel':False,'operators':['MatMul'],
                                    'observed_quantized_operator_types':sorted(changed_ops.intersection({'MatMulInteger','QLinearMatMul','DynamicQuantizeLinear'}))},
                    'fp32_metrics':fp32,'int8_metrics':int8,'decision':gate,
                    'limits':['Measured graph size, NOT necessarily complete package or RAM footprint.',
                              'INT8 can be smaller yet slower or less accurate; gate failures are expected possibilities.',
                              'Development split used for candidate selection; requires NEW evaluation before export.',
                              'No independent external quality certification, production approval, FLOPs, energy or cloud billing.']}
            report_path=out_dir/'report.json'
            report_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
            return report_path,result
    except Exception:
        # Failed trials do not leave misleading half-built INT8 model artifacts.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        raise
