"""NIMBO M4: review-only, offline-reloadable ONNX classifier bundle.

M2's exact graph, M3's gated report, the pinned tokenizer/config, independent
unzip+subprocess inference, and checksum validation. NOT production approval.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from app.nimbo_heldout_validation_v1 import load_heldout, preprocess_text, read_m2_candidate
from app.nimbo_model_intake_v1 import inspect_public_classifier, resolve_labels
from app.nimbo_onnx_candidate_v1 import digest

EXPECTED_M2_STATUS = 'EXPERIMENT_ONLY_NOT_EXPORTED'
EXPECTED_M3_STATUS = 'HELDOUT_VALIDATION_COMPLETED_REVIEW_ONLY'
MAX_PACKAGE_BYTES = 950 * 1024 * 1024
MAX_BUNDLE_FILES = 24


def check_review(m3_path: str | Path, dataset: str | Path, *, root: str | Path='.') -> tuple[dict, dict, Path, list[dict], str]:
    """Bind M2 graph, immutable model identity, M3 holdout, and actual dataset."""
    project = Path(root).expanduser().resolve(strict=True)
    p = Path(m3_path).expanduser()
    if p.is_symlink():
        raise ValueError('Symlink M3 report refused')
    p = p.resolve(strict=True)
    expected_root = (project/'results'/'auto_optimizer_heldout_v1').resolve()
    if expected_root not in p.parents or p.suffix != '.json':
        raise ValueError('M3 review report must be under original project heldout results')
    m3 = json.loads(p.read_text(encoding='utf8'))
    if m3.get('schema_version') != 'nimbo_heldout_validation_v1' or m3.get('status') != EXPECTED_M3_STATUS:
        raise ValueError('M3 report missing or not a completed heldout review')
    gate = m3.get('decision') or {}
    if not all(gate.get(k) is True for k in ('heldout_gates_passed','quality_pass','performance_pass')) or gate.get('decision') != 'HELDOUT_LOCAL_PASS_REQUIRES_PACKAGE_RELOAD':
        raise ValueError('Cannot package a candidate which failed M3 gates')
    if gate.get('repeatability_mismatches') != 0 or gate.get('prediction_agreement') != 1.0 or gate.get('accuracy_drop') != 0.0:
        raise ValueError('M3 consistency checks do not meet strict review policy')
    if m3.get('heldout_unique_examples',0) < 200 or m3.get('rounds',0) < 3:
        raise ValueError('M3 evaluation sample count or rounds insufficient')
    rows, heldout_sha, provenance = load_heldout(dataset)
    if len(rows) != m3.get('heldout_unique_examples') or m3.get('heldout_dataset_sha256') != heldout_sha:
        raise ValueError('M3 heldout dataset identity mismatch')
    if provenance != m3.get('heldout_dataset_provenance'):
        raise ValueError('Dataset provenance differs from the M3 validation')
    model_info = m3.get('model') or {}
    model_id, revision = model_info.get('model_id'), model_info.get('resolved_revision')
    if not isinstance(model_id,str) or not isinstance(revision,str):
        raise ValueError('M3 missing pinned model identity')
    cpu_threads, max_length = m3.get('cpu_threads_each_path'), m3.get('max_length')
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8 or type(max_length) is not int or not 32 <= max_length <= 256:
        raise ValueError('Invalid model execution limits')
    source_name = m3.get('source_candidate_report')
    if not isinstance(source_name,str):
        raise ValueError('M3 missing M2 source')
    m2_path = Path(source_name).expanduser().resolve(strict=True)
    m2, graph = read_m2_candidate(m2_path,root=project,model_id=model_id,sha=revision,
         heldout_sha=heldout_sha,threads=cpu_threads,max_length=max_length)
    if (m3.get('source_candidate_report') != str(m2_path) or
        m3.get('source_graph_sha256') != m2['artifact']['sha256'] or
        m3.get('source_m2_dataset_sha256') != m2['dataset_sha256']):
        raise ValueError('M3 review is not linked to this exact M2 graph and source dataset')
    if m3.get('preprocessing') not in ('identity','cardiff_tweet'):
        raise ValueError('Unexpected input preprocessing policy')
    for key in ('baseline','candidate'):
        if not isinstance(m3.get(key),dict) or m3[key].get('accuracy') is None:
            raise ValueError('M3 missing measured accuracy')
        if m3[key].get('n_forward_calls') != len(rows)*m3['rounds']:
            raise ValueError('M3 measurement counts do not match heldout rows and rounds')
    # M2/M3 reports are local evidence, not cryptographically signed attestations.
    return m3,m2,graph,rows,heldout_sha


def _sha_predictions(preds: list[int]) -> str:
    return hashlib.sha256(','.join(map(str,preds)).encode('ascii')).hexdigest()


def _build_session(package_dir: Path, threads: int):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(package_dir/'model.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('Required CPUExecutionProvider unavailable')
    names = tuple(i.name for i in session.get_inputs())
    if 'input_ids' not in names or any(i not in ('input_ids','attention_mask','token_type_ids') for i in names):
        raise ValueError('Unexpected ONNX input signature')
    if len(session.get_outputs()) != 1:
        raise ValueError('Unexpected ONNX output signature')
    return session,names


def _predict_bundle(package_dir: Path, rows: list[dict], *, threads: int, length: int,
                    preprocess: str, num_labels: int) -> list[int]:
    import numpy as np
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(package_dir),local_files_only=True,trust_remote_code=False)
    session,names = _build_session(package_dir,threads)
    out = []
    for row in rows:
        tokens=tokenizer(preprocess_text(row['text'],preprocess),truncation=True,max_length=length,return_tensors='np')
        if any(n not in tokens for n in names):
            raise ValueError('Saved tokenizer cannot satisfy ONNX input signature')
        values={n:np.asarray(tokens[n],dtype=np.int64) for n in names}
        logits=np.asarray(session.run(None,values)[0])
        if logits.shape!=(1,num_labels) or not np.isfinite(logits).all():
            raise ValueError('Exported model returned invalid logits')
        out.append(int(np.argmax(logits,axis=1)[0]))
    return out


def _hashes(folder: Path) -> dict[str,str]:
    files = sorted(f for f in folder.iterdir() if f.is_file())
    if any(f.is_symlink() for f in folder.iterdir()) or any(f.is_dir() for f in folder.iterdir()):
        raise ValueError('Unsafe nested path or symbolic link in model package')
    if len(files)>MAX_BUNDLE_FILES or any(f.stat().st_size>MAX_PACKAGE_BYTES for f in files):
        raise ValueError('Unexpected package file count or size')
    return {f.name: digest(f) for f in files if f.name!='manifest.json'}


def verify_bundle_files(folder: Path, manifest: dict) -> None:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError('Missing or symlinked model package directory')
    file_hashes=manifest.get('files')
    if not isinstance(file_hashes,dict) or not file_hashes or len(file_hashes)>MAX_BUNDLE_FILES:
        raise ValueError('Missing package file hashes')
    if file_hashes != _hashes(folder):
        raise ValueError('Model package files or checksums differ from manifest')
    if not {'model.onnx','config.json','tokenizer_config.json','run_model.py','README.md','evidence.json'}.issubset(file_hashes):
        raise ValueError('Missing required runnable package files')


def verify_package(package: str | Path, dataset: str | Path) -> dict:
    """No network; independently reload the extracted bundle on actual data."""
    source = Path(package).expanduser()
    if source.is_symlink():
        raise ValueError('Symlinked model package refused')
    folder = source.resolve(strict=True)
    if (folder/'manifest.json').is_symlink():
        raise ValueError('Symlinked model manifest refused')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    if manifest.get('schema_version')!='nimbo_optimized_package_v1' or manifest.get('review_status')!='LOCAL_REVIEW_ONLY_NOT_DEPLOYED':
        raise ValueError('Not a valid NIMBO local review package')
    verify_bundle_files(folder,manifest)
    rows,sha,_=load_heldout(dataset)
    if sha!=manifest.get('heldout_dataset_sha256') or len(rows)!=manifest.get('heldout_examples'):
        raise ValueError('External dataset does not match reviewed sample set')
    if len(rows)<200:
        raise ValueError('Insufficient package verification samples')
    spec=manifest.get('model') or {}
    names=spec.get('id2label') or {}
    if not isinstance(names,dict) or len(names)!=spec.get('num_labels'):
        raise ValueError('Invalid class mapping in model manifest')
    mapping={int(k):str(v) for k,v in names.items()}
    expected=resolve_labels(rows,mapping)
    if any(v is None for v in expected):
        raise ValueError('All reference labels required')
    preds=_predict_bundle(folder,rows,threads=manifest['cpu_threads'],length=manifest['max_length'],
                          preprocess=manifest['preprocessing'],num_labels=spec['num_labels'])
    acc=round(sum(int(p==e) for p,e in zip(preds,expected))/len(rows),6)
    if _sha_predictions(preds)!=manifest.get('validation_predictions_sha256'):
        raise ValueError('Offline model predictions do not reproduce packaging-time predictions')
    if abs(acc-manifest['expected_candidate_accuracy'])>1e-6:
        raise ValueError('Reloaded model accuracy differs from M3 results')
    return {'status':'OFFLINE_BUNDLE_RELOAD_VERIFIED_REVIEW_ONLY','samples':len(rows),
            'accuracy':acc,'model_graph_sha256':manifest['files']['model.onnx'],
            'prediction_fingerprint':_sha_predictions(preds),'eligible_for_deployment':False}


def _runner_source() -> str:
    return '''#!/usr/bin/env python3
"""Run a self-contained NIMBO ONNX classifier locally (no network required)."""
import argparse,hashlib,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description="Run NIMBO ONNX FP32 model locally")
    p.add_argument('--text',required=True)
    args=p.parse_args()
    folder=Path(__file__).resolve().parent
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    files=manifest.get('files',{})
    names={f.name for f in folder.iterdir() if f.is_file() and f.name!='manifest.json'}
    if names!=set(files) or any(f.is_symlink() for f in folder.iterdir()):
        raise RuntimeError('Package file list changed')
    for name,sha in files.items():
        h=hashlib.sha256()
        with (folder/name).open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
        if h.hexdigest()!=sha:raise RuntimeError('File integrity check failed: '+name)
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=int(manifest['cpu_threads'])
    opts.inter_op_num_threads=1
    sess=ort.InferenceSession(str(folder/'model.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    if sess.get_providers()!=['CPUExecutionProvider']:raise RuntimeError('CPU execution unavailable')
    tokenizer=AutoTokenizer.from_pretrained(str(folder),trust_remote_code=False,local_files_only=True)
    text=args.text
    if manifest['preprocessing']=='cardiff_tweet':
        text=' '.join('@user' if w.startswith('@') and len(w)>1 else 'http' if w.startswith('http') else w for w in text.split(' '))
    tokens=tokenizer(text,max_length=int(manifest['max_length']),truncation=True,return_tensors='np')
    names=[x.name for x in sess.get_inputs()]
    feeds={n:np.asarray(tokens[n],dtype=np.int64) for n in names}
    logits=np.asarray(sess.run(None,feeds)[0])
    classes=manifest['model']['id2label']
    index=int(logits.argmax(axis=1)[0])
    print(json.dumps({'label':classes[str(index)],'class_id':index,'review_only':True},ensure_ascii=False))

if __name__=='__main__':
    main()
'''


def package_model(m3_report: str | Path, dataset: str | Path, *, root: str | Path='.',
                  output_root: str | Path='results/auto_optimizer_packages_v1',
                  runner_executable: str | None=None) -> tuple[Path,dict]:
    """Package same hash-pinned M2 ONNX only after M3 pass + fresh offline reload."""
    m3,m2,graph,rows,heldout_sha=check_review(m3_report,dataset,root=root)
    info=m3['model']
    spec=inspect_public_classifier(info['model_id'],revision=info['resolved_revision'])
    if spec.model_id!=info['model_id'] or spec.resolved_revision!=info['resolved_revision'] or spec.num_labels!=info['num_labels']:
        raise ValueError('Pinned model metadata changed since M3')
    expected_mappings={str(k):v for k,v in spec.id2label.items()}
    if expected_mappings!=info.get('id2label'):
        raise ValueError('Classification labels differ from M3')
    root_out=Path(output_root).expanduser().resolve()
    # Do not create arbitrary user-selected destination paths.
    if root_out != (Path(root).expanduser().resolve(strict=True)/'results'/'auto_optimizer_packages_v1').resolve():
        raise ValueError('Package output must be the dedicated project results folder')
    root_out.mkdir(parents=True,exist_ok=True)
    import uuid
    name='nimbo_onnx_review_'+uuid.uuid4().hex[:12]
    dest=root_out/name
    zip_file=root_out/(name+'.zip')
    if dest.exists() or zip_file.exists():
        raise FileExistsError('Package destination already exists')
    size=graph.stat().st_size
    if size>MAX_PACKAGE_BYTES:
        raise ValueError('ONNX graph too large')
    if shutil.disk_usage(root_out).free < size*3 + 200*1024*1024:
        raise RuntimeError('Not enough disk space to stage, ZIP, and verify the model; need about 3x graph size plus 200 MiB')
    staging=Path(tempfile.mkdtemp(prefix='.m4_staging_',dir=root_out))
    archive_created=False
    try:
        from transformers import AutoTokenizer,AutoConfig
        tokenizer=AutoTokenizer.from_pretrained(spec.model_id,revision=spec.resolved_revision,trust_remote_code=False)
        config=AutoConfig.from_pretrained(spec.model_id,revision=spec.resolved_revision,trust_remote_code=False)
        if config.model_type!=spec.architecture or config.num_labels!=spec.num_labels:
            raise ValueError('Saved config and M3 model type mismatch')
        tokenizer.save_pretrained(str(staging))
        config.save_pretrained(str(staging))
        shutil.copy2(graph,staging/'model.onnx')
        if digest(staging/'model.onnx')!=m2['artifact']['sha256']:
            raise ValueError('Copied model ONNX does not match M2/M3 source')
        (staging/'run_model.py').write_text(_runner_source(),encoding='utf8')
        (staging/'README.md').write_text(
            '# NIMBO ONNX FP32 classifier — LOCAL REVIEW ONLY\n\n'
            'Extract the ZIP, install `numpy`, `transformers` and `onnxruntime`, then run:\n\n'
            '`HF_HUB_OFFLINE=1 python run_model.py --text "Example text"`\n\n'
            'Files: model.onnx, tokenizer, config.json, manifest.json, evidence.json.\n'
            'Local heldout evidence is review-only, NOT production approval or independent external certification.\n',encoding='utf8')
        evidence={'source_m3_report_sha256':digest(Path(m3_report).expanduser().resolve(strict=True)),
                  'source_m2_report_sha256':digest(Path(m3['source_candidate_report'])),
                  'source_graph_sha256':m2['artifact']['sha256'],
                  'heldout_dataset_sha256':heldout_sha,
                  'heldout_unique_examples':m3['heldout_unique_examples'],
                  'preprocessing':m3['preprocessing'],
                  'baseline':m3['baseline'],'candidate':m3['candidate'],'decision':m3['decision'],
                  'note':'Model/source training-independent validation not established; review only; no energy, billing or model size reduction evidence.'}
        (staging/'evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        predictions=_predict_bundle(staging,rows,threads=m3['cpu_threads_each_path'],
             length=m3['max_length'],preprocess=m3['preprocessing'],num_labels=spec.num_labels)
        from app.nimbo_model_intake_v1 import resolve_labels
        labels=resolve_labels(rows,spec.id2label)
        accuracy=round(sum(int(p==y) for p,y in zip(predictions,labels))/len(rows),6)
        if abs(accuracy-m3['candidate']['accuracy'])>1e-6:
            raise ValueError('Staged package does not reproduce M3 candidate accuracy')
        manifest={'schema_version':'nimbo_optimized_package_v1',
                  'review_status':'LOCAL_REVIEW_ONLY_NOT_DEPLOYED',
                  'model':{'model_id':spec.model_id,'resolved_revision':spec.resolved_revision,
                           'architecture':spec.architecture,'num_labels':spec.num_labels,'id2label':expected_mappings},
                  'source_m2_graph_sha256':m2['artifact']['sha256'],
                  'source_m3_report_sha256':evidence['source_m3_report_sha256'],
                  'heldout_dataset_sha256':heldout_sha,'heldout_examples':len(rows),
                  'cpu_threads':m3['cpu_threads_each_path'],'max_length':m3['max_length'],
                  'preprocessing':m3['preprocessing'],
                  'expected_candidate_accuracy':m3['candidate']['accuracy'],
                  'validation_predictions_sha256':_sha_predictions(predictions),
                  'files':_hashes(staging),
                  'notes':['Same FP32 weights in ONNX format; smaller on-disk model is NOT established.',
                           'Deployment requires separate production security and workload validation.',
                           'No guarantee source model never saw TweetEval data.']}
        verify_bundle_files(staging,manifest)
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        with zipfile.ZipFile(zip_file,'x') as archive:
            for item in sorted(staging.iterdir()):
                kind=zipfile.ZIP_STORED if item.name=='model.onnx' else zipfile.ZIP_DEFLATED
                archive.write(item,arcname=f'{name}/{item.name}',compress_type=kind)
        archive_created=True
        # Make a fresh, separate extracted workspace and python process; no HF network.
        with tempfile.TemporaryDirectory(prefix='.m4_verify_',dir=root_out) as temp:
            location=Path(temp)
            with zipfile.ZipFile(zip_file) as archive:
                names=archive.namelist()
                if len(names)>MAX_BUNDLE_FILES+1 or any(v not in {name+'/'+s for s in (*manifest['files'],'manifest.json')} for v in names):
                    raise RuntimeError('Unexpected archive member: refuse to extract')
                archive.extractall(location)
            package_dir=location/name
            env=os.environ.copy()
            env.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'})
            args=[runner_executable or sys.executable,'-m','tools.verify_nimbo_package_v1','--package',str(package_dir),'--dataset',str(Path(dataset).expanduser().resolve(strict=True))]
            completed=subprocess.run(args,cwd=str(Path(root).expanduser().resolve(strict=True)),
                                     capture_output=True,text=True,env=env,timeout=300)
            if completed.returncode!=0 or 'OFFLINE_BUNDLE_RELOAD_VERIFIED_REVIEW_ONLY' not in completed.stdout:
                raise RuntimeError('Offline extracted-package reload failed: '+(completed.stderr+completed.stdout)[-1500:])
            # Verify packaged executable from extracted ZIP also boots offline.
            smoke=subprocess.run([runner_executable or sys.executable,str(package_dir/'run_model.py'),
                                   '--text',rows[0]['text']],capture_output=True,text=True,env=env,timeout=120)
            if smoke.returncode!=0 or '"review_only": true' not in smoke.stdout.lower():
                raise RuntimeError('Offline standalone runner failed: '+(smoke.stderr+smoke.stdout)[-1500:])
        shutil.rmtree(staging)
        summary={'status':'RUNNABLE_PACKAGE_VERIFIED_REVIEW_ONLY','archive':str(zip_file),
                 'archive_sha256':digest(zip_file),'source_graph_sha256':m2['artifact']['sha256'],
                 'source_m3_report_sha256':evidence['source_m3_report_sha256'],
                 'heldout_examples':len(rows),'reload_accuracy':accuracy,
                 'verified_offline_after_extract':True,
                 'eligible_for_deployment':False,
                 'measured_smaller_model':False}
        return zip_file,summary
    except Exception:
        if staging.exists():shutil.rmtree(staging)
        if zip_file.exists():zip_file.unlink()
        raise
