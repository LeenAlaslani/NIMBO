"""M7 API and worker tests use a MOCK injection. No real RoBERTa executed."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from app.nimbo_auto_pipeline_jobs_v1 import (
    validate_request, submit, read_job, read_events, run_worker,
    package_for_download, _atomic_json, _event, job_folder,
)


def setup(root):
    (root/'results'/'auto_optimizer_job_inputs_v1').mkdir(parents=True,exist_ok=True)
    (root/'results'/'auto_optimizer_heldout_v1').mkdir(parents=True,exist_ok=True)
    (root/'results'/'auto_optimizer_jobs_v1').mkdir(parents=True,exist_ok=True)
    (root/'results'/'auto_optimizer_packages_v1').mkdir(parents=True,exist_ok=True)
    pilot=root/'results'/'auto_optimizer_job_inputs_v1'/'pilot.csv'
    pilot.write_text('text,label\nhello,POSITIVE\nawful,NEGATIVE\n')
    hold=root/'results'/'auto_optimizer_heldout_v1'/'heldout.csv'
    hold.write_text('text,label\n'+''.join(f'example {i},{i%2}\n' for i in range(200)))
    req={'model_id':'team/roberta','revision':'a'*40,'pilot_file':'pilot.csv',
         'heldout_file':'heldout.csv','preprocess':'identity','objective':'speed'}
    return req


def fake_spawn(*args,**kwargs):
    assert args[0][2]=='tools.run_nimbo_auto_pipeline_job_v1'
    assert kwargs['start_new_session'] is True
    assert kwargs['cwd']
    return SimpleNamespace(pid=12345)


def create(root):
    return submit(root,setup(root),spawn=fake_spawn)


def test_preflight_rejects_unsafe_filename_and_unknown_objectives(tmp_path):
    req=setup(tmp_path)
    assert validate_request(tmp_path,req)['heldout_rows']==200
    for name in ('../pilot.csv','/tmp/foo.csv','pilot.txt'):
        broken=dict(req,pilot_file=name)
        with pytest.raises(ValueError):validate_request(tmp_path,broken)
    with pytest.raises(ValueError,match='speed'):
        validate_request(tmp_path,dict(req,objective='size'))
    with pytest.raises(ValueError):validate_request(tmp_path,dict(req,extra=True))


def test_file_tampering_rejected_and_status_error(tmp_path):
    job=create(tmp_path)
    folder=job_folder(tmp_path,job['job_id'])
    (tmp_path/'results'/'auto_optimizer_job_inputs_v1'/'pilot.csv').write_text('text,label\nmodified,0\nchanged,1\n')
    result=run_worker(tmp_path,job['job_id'])
    assert result['state']=='FAILED' and result['stage']=='error'


def test_single_active_job_and_event_sequence(tmp_path,monkeypatch):
    import app.nimbo_auto_pipeline_jobs_v1 as mod
    monkeypatch.setattr(mod,'_process_alive',lambda pid:True)
    first=create(tmp_path)
    events=read_events(tmp_path,first['job_id'])
    assert events['events'][0]['seq']==1
    with pytest.raises(RuntimeError,match='already running'):
        submit(tmp_path,setup(tmp_path),spawn=fake_spawn)
    with pytest.raises(ValueError):read_events(tmp_path, first['job_id'],after=-1)
    with pytest.raises(ValueError):job_folder(tmp_path,'../../etc')


def test_worker_success_and_real_stage_events_mocked(tmp_path):
    job=create(tmp_path)
    folder=job_folder(tmp_path,job['job_id'])
    def baseline(*args,**kw):
        assert kw['revision']=='a'*40
        return tmp_path/'m1.json',{'metrics':{'accuracy':.71,'mean_warm_forward_ms':44.0}}
    def candidate(*args,**kw):
        assert kw['baseline_report']==tmp_path/'m1.json'
        return tmp_path/'m2.json',{'decision':{'local_gates_passed':True,'decision':'EXPLORATORY_PASS_NOT_APPROVED'}}
    def heldout(*args,**kw):
        return tmp_path/'m3.json',{'decision':{'heldout_gates_passed':True,'decision':'HELDOUT_LOCAL_PASS_REQUIRES_PACKAGE_RELOAD'},'candidate':{'accuracy':.71}}
    p=tmp_path/'results'/'auto_optimizer_packages_v1'/'model.zip';p.write_bytes(b'fake_package')
    def packager(*args,**kw):
        return p,{'status':'RUNNABLE_PACKAGE_VERIFIED_REVIEW_ONLY',
                  'verified_offline_after_extract':True,'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                  'reload_accuracy':.71}
    result=run_worker(tmp_path,job['job_id'],baseline=baseline,candidate=candidate,heldout=heldout,packager=packager)
    assert result['state']=='COMPLETED_REVIEW_ONLY'
    assert package_for_download(tmp_path,job['job_id'])==p
    p.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checksum'):
        package_for_download(tmp_path,job['job_id'])
    stages=[e['stage'] for e in read_events(tmp_path,job['job_id'])['events']]
    assert stages==['queued','analyze','analyze','optimize','optimize','verify','verify','deliver','deliver']
    assert json.loads((folder/'receipt.json').read_text())['eligible_for_deployment'] is False


def test_worker_rejects_onnx_without_package(tmp_path):
    job=create(tmp_path)
    def baseline(*a,**k):return tmp_path/'m1',{'metrics':{'accuracy':.7,'mean_warm_forward_ms':40}}
    def candidate(*a,**k):return tmp_path/'m2',{'decision':{'local_gates_passed':False,'decision':'REJECTED_LOCAL_GATES'}}
    result=run_worker(tmp_path,job['job_id'],baseline=baseline,candidate=candidate)
    assert result['state']=='COMPLETED_REJECTED'
    assert result['decision']=='SELECT_ORIGINAL_NO_ONNX_PACKAGE'
    with pytest.raises(ValueError):package_for_download(tmp_path,job['job_id'])


def test_worker_fails_closed_on_packager_misreport(tmp_path):
    job=create(tmp_path)
    def baseline(*a,**k):return tmp_path/'m1',{'metrics':{'accuracy':.7,'mean_warm_forward_ms':40}}
    def candidate(*a,**k):return tmp_path/'m2',{'decision':{'local_gates_passed':True,'decision':'LOCAL_PASS_REQUIRES_HELDOUT_AND_RELOAD'}}
    def heldout(*a,**k):return tmp_path/'m3',{'decision':{'heldout_gates_passed':True,'decision':'OK'},'candidate':{'accuracy':.7}}
    def packager(*a,**k):return tmp_path/'bad.zip',{'status':'NOT_VERIFIED'}
    out=run_worker(tmp_path,job['job_id'],baseline=baseline,candidate=candidate,heldout=heldout,packager=packager)
    assert out['state']=='FAILED' and out['decision']=='EXECUTION_FAILED'


def test_status_marks_exited_worker_interrupted(tmp_path,monkeypatch):
    job=create(tmp_path)
    import app.nimbo_auto_pipeline_jobs_v1 as mod
    monkeypatch.setattr(mod,'_process_alive',lambda pid:False)
    s=read_job(tmp_path,job['job_id'])
    assert s['state']=='INTERRUPTED'


def test_api_routes_without_touching_running_server(tmp_path,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import app.nimbo_auto_jobs_api_v1 as api
    setup(tmp_path)
    monkeypatch.setattr(api,'_local',lambda request: tmp_path)
    monkeypatch.setattr(api,'submit',lambda root,request:{'job_id':'a'*32,'state':'QUEUED'})
    app=FastAPI();app.include_router(api.router)
    client=TestClient(app)
    payload={'model_id':'team/model','revision':'a'*40,'pilot_file':'pilot.csv','heldout_file':'heldout.csv'}
    res=client.post('/nimbo/auto/jobs',json=payload)
    assert res.status_code==202 and res.json()['state']=='QUEUED'
    assert client.post('/nimbo/auto/jobs',json=dict(payload,objective='size')).status_code==202  # mocked submit
    assert client.post('/nimbo/auto/jobs',json=dict(payload,revision='main')).status_code==422


def test_api_refuses_remote_client(tmp_path,monkeypatch):
    from fastapi import HTTPException
    import app.nimbo_auto_jobs_api_v1 as api
    with pytest.raises(HTTPException) as exc:
        api._local(SimpleNamespace(client=SimpleNamespace(host='192.0.2.44')))
    assert exc.value.status_code == 403


def test_preflight_rejects_changed_or_unlabeled_inputs(tmp_path):
    req=setup(tmp_path)
    pilot=tmp_path/'results'/'auto_optimizer_job_inputs_v1'/'pilot.csv'
    pilot.write_text('text,label\nhello,\nawful,NEGATIVE\n')
    with pytest.raises(ValueError,match='fully labeled'):
        validate_request(tmp_path,req)
    pilot.write_text('text,label\nhello,POSITIVE\nawful,NEGATIVE\n')
    req['revision']='main'
    with pytest.raises(ValueError):validate_request(tmp_path,req)


def test_source_symlink_forbidden(tmp_path):
    req=setup(tmp_path)
    pilot=tmp_path/'results'/'auto_optimizer_job_inputs_v1'/'pilot.csv'
    actual=tmp_path/'other.csv'
    pilot.rename(actual)
    pilot.symlink_to(actual)
    with pytest.raises(ValueError,match='symlinked'):
        validate_request(tmp_path,req)


def test_heldout_failure_stops_before_packaging(tmp_path):
    job=create(tmp_path)
    def baseline(*a,**k):return tmp_path/'m1',{'metrics':{'accuracy':.7,'mean_warm_forward_ms':40}}
    def candidate(*a,**k):return tmp_path/'m2',{'decision':{'local_gates_passed':True,'decision':'EXPLORATORY_PASS_NOT_APPROVED'}}
    def heldout(*a,**k):return tmp_path/'m3',{'decision':{'heldout_gates_passed':False,'decision':'REJECTED_HELDOUT_GATES'}}
    def never_package(*a,**k):raise AssertionError('rejected model must not be packaged')
    status=run_worker(tmp_path,job['job_id'],baseline=baseline,candidate=candidate,heldout=heldout,packager=never_package)
    assert status['state']=='COMPLETED_REJECTED'
    assert status['decision']=='SELECT_ORIGINAL_NO_ONNX_PACKAGE'
