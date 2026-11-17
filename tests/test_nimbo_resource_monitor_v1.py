"""M12.4 resource monitoring: no model download or ONNX computation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import nimbo_resource_monitor_v1 as m


def mkjob(root: Path, kind="agentic", state="RUNNING", job_id="a"*32):
    base = "results/agentic_jobs_v1" if kind == "agentic" else "results/auto_optimizer_jobs_v1"
    directory = root / base / job_id
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(json.dumps({"state": state, "created_at": "2026-11-15T11:00:00+00:00"}))
    return directory


def test_historical_never_reports_live_cpu(tmp_path):
    d = mkjob(tmp_path, state="COMPLETED_REVIEW_ONLY")
    result = m.sample_job_resources(tmp_path, "a"*32, "agentic")
    assert result["monitor_state"] == "finished"
    assert result["job_cpu_percent_one_core"] is None
    assert result["gpu_status"] == "not_instrumented"


def test_strict_identity_and_symlink(tmp_path):
    mkjob(tmp_path)
    with pytest.raises(ValueError):
        m.sample_job_resources(tmp_path, "../../etc", "agentic")
    with pytest.raises(FileNotFoundError):
        m.sample_job_resources(tmp_path, "b"*32, "agentic")
    link = tmp_path / "results/auto_optimizer_jobs_v1" / ("c"*32)
    link.parent.mkdir(parents=True)
    link.symlink_to(tmp_path / "results/agentic_jobs_v1" / ("a"*32), target_is_directory=True)
    with pytest.raises(ValueError):
        m.sample_job_resources(tmp_path, "c"*32, "auto")


def test_legacy_job_fallback(tmp_path):
    mkjob(tmp_path,kind="auto",state="COMPLETED_REJECTED")
    result=m.sample_job_resources(tmp_path,"a"*32,"agentic")
    assert result["job_kind"]=="auto"
    assert result["monitor_state"]=="finished"


def test_missing_psutil_not_zero(tmp_path,monkeypatch):
    d=mkjob(tmp_path)
    (d/"launcher.json").write_text(json.dumps({"pid": 12345}))
    monkeypatch.setattr(m,"psutil",None)
    r=m.sample_job_resources(tmp_path,"a"*32,"agentic")
    assert r["monitor_state"]=="unavailable"
    assert r["job_cpu_percent_one_core"] is None


def test_process_counter_multi_core_and_reset():
    assert m._cpu_delta_percent({(1,12):2}, {(1,12):5}, 2)==150.0
    assert m._cpu_delta_percent({}, {(1,12):5}, 2) is None
    assert m._cpu_delta_percent({(1,12):2}, {(1,13):5}, 2) is None


def test_live_samples_record_real_cpu_deltas(tmp_path, monkeypatch):
    directory=mkjob(tmp_path)
    (directory/"launcher.json").write_text(json.dumps({"pid":1234}))
    class Times:
        def __init__(self,user): self.user=user;self.system=0
    class Proc:
        pid=1234
        def create_time(self):return 1794740401.0
        def children(self,recursive=False):return []
        def is_running(self):return True
        def status(self):return "running"
        def cpu_times(self):return Times(calls[0])
        def memory_info(self):return SimpleNamespace(rss=128*1024*1024)
    calls=[10.0]
    process=Proc()
    fake=SimpleNamespace(
        Process=lambda pid:process,
        process_iter=lambda **kwargs:[],
        virtual_memory=lambda:SimpleNamespace(used=256*1024*1024,total=1024*1024*1024,percent=25),
        cpu_times=lambda:SimpleNamespace(_asdict=lambda:{"user":10.0,"idle":5.0}),
        STATUS_ZOMBIE="zombie",NoSuchProcess=type("NoSuchProcess",(Exception,),{}),
        AccessDenied=type("AccessDenied",(Exception,),{}),ZombieProcess=type("ZombieProcess",(Exception,),{})
    )
    monkeypatch.setattr(m,"psutil",fake)
    clock=[100.0]
    monkeypatch.setattr(m.time,"monotonic",lambda:clock[0])
    m._SAMPLES.clear()
    first=m.sample_job_resources(tmp_path,"a"*32,"agentic")
    assert first["monitor_state"]=="live" and first["job_cpu_percent_one_core"] is None
    clock[0]=102.0
    calls[0]=13.0
    second=m.sample_job_resources(tmp_path,"a"*32,"agentic")
    assert second["job_cpu_percent_one_core"]==150.0
    assert second["job_rss_mib"]==128
    assert second["observed_peak_job_rss_mib"]==128
    assert second["host_ram_percent"]==25
    assert second["sample_count"]==2
    m._SAMPLES.clear()


def test_reused_pid_is_not_sampled(tmp_path, monkeypatch):
    d=mkjob(tmp_path)
    (d/"launcher.json").write_text(json.dumps({"pid":1234,"at":"2026-11-15T12:00:00+00:00"}))
    class OldProc:
        def create_time(self): return 1000000000.0
    monkeypatch.setattr(m,"psutil",SimpleNamespace(Process=lambda pid:OldProc(),NoSuchProcess=Exception,AccessDenied=Exception,ZombieProcess=Exception))
    r=m.sample_job_resources(tmp_path,"a"*32,"agentic")
    assert r["monitor_state"]=="unavailable"
    assert r["job_rss_mib"] is None
