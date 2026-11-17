"""M12.4: read-only, loopback-only, process-scoped LIVE resource samples.

This is viewer-triggered telemetry, not an inference benchmark or durable job evidence.
CPU 100% means one logical CPU core. RSS sums can double-count shared pages.
A missing dependency/process never becomes a fabricated zero measurement.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:  # Deliberate graceful degradation; install in the API venv.
    psutil = None

_ID = re.compile(r"^[0-9a-f]{32}$")
_ACTIVE = frozenset({"QUEUED", "RUNNING"})
_LOCK = threading.Lock()
_SAMPLES: dict[tuple[str, str], dict] = {}
_MAX_CACHED = 128


def _job_dir(root: Path, job_id: str, preferred: str) -> tuple[Path, str]:
    if not _ID.fullmatch(job_id):
        raise ValueError("Invalid job ID")
    if preferred not in ("agentic", "auto"):
        raise ValueError("Unknown job kind")
    choices = (("agentic", "results/agentic_jobs_v1"), ("auto", "results/auto_optimizer_jobs_v1"))
    if preferred == "auto":
        choices = choices[1:]
    for kind, relative in choices:
        parent = root / relative
        path = parent / job_id
        if path.is_symlink():
            raise ValueError("Job path is a symlink")
        if path.is_dir():
            return path, kind
    raise FileNotFoundError("Unknown job")


def _load_json(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("Symlinked job metadata is not allowed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid job metadata")
    return value


def _time_seconds(value) -> float | None:
    try:
        if not isinstance(value, str):
            return None
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return None
        return dt.timestamp()
    except ValueError:
        return None


def _process_record(process) -> tuple[tuple[int, float], float, int] | None:
    try:
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        ct = process.create_time()
        cpu = process.cpu_times()
        rss = process.memory_info().rss
        if not (math.isfinite(ct) and math.isfinite(cpu.user + cpu.system)) or rss < 0:
            return None
        return ((process.pid, ct), cpu.user + cpu.system, rss)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError, ValueError):
        return None


def _cpu_delta_percent(before: dict, after: dict, elapsed: float) -> float | None:
    if not before or elapsed <= 0:
        return None
    overlaps = [max(0.0, cpu - before[key]) for key, cpu in after.items() if key in before]
    if not overlaps:
        return None
    return round(100 * sum(overlaps) / elapsed, 1)


def _host_cpu(before, after) -> float | None:
    if before is None or after is None:
        return None
    try:
        b = before._asdict()
        a = after._asdict()
        total = sum(a.values()) - sum(b.values())
        idle = a.get("idle", 0) - b.get("idle", 0)
        idle += a.get("iowait", 0) - b.get("iowait", 0)
        if total <= 0:
            return None
        return round(max(0.0, min(100.0, 100 * (total - idle) / total)), 1)
    except (AttributeError, ValueError, TypeError):
        return None


def _read_processes(root_pid: int, launched_at: float | None):
    try:
        parent = psutil.Process(root_pid)
        parent_birth = parent.create_time()
        # A recycled PID must never cause unrelated system processes to be sampled.
        if launched_at is not None and parent_birth < launched_at - 6:
            return None
        group = [parent] + parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        return None
    samples = {}
    rss_sum = 0
    for process in group:
        result = _process_record(process)
        if result is not None:
            identity, cpu, rss = result
            samples[identity] = cpu
            rss_sum += rss
    return (samples, rss_sum, len(samples)) if samples else None


def _ollama_processes(excluded: set[int]):
    samples = {}
    rss_sum = 0
    try:
        iterator = psutil.process_iter(attrs=["name"], ad_value=None)
        for proc in iterator:
            try:
                name = str(proc.info.get("name") or "").casefold()
                if proc.pid in excluded or not name.startswith("ollama"):
                    continue
                result = _process_record(proc)
                if result is not None:
                    identity, cpu, rss = result
                    samples[identity] = cpu
                    rss_sum += rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        pass
    return samples, rss_sum


def sample_job_resources(root: Path, job_id: str, preferred: str) -> dict:
    """Return one bounded resource snapshot. Never launches a worker or infers."""
    root = Path(root).resolve(strict=True)
    directory, kind = _job_dir(root, job_id, preferred)
    status = _load_json(directory / "status.json")
    state = str(status.get("state", "UNKNOWN"))
    key = (kind, job_id)
    result = {
        "schema_version": "nimbo_resource_sample_v1",
        "job_id": job_id, "job_kind": kind, "job_state": state,
        "sampled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "monitor_state": "waiting", "scope": "job_worker_process_tree",
        "job_cpu_percent_one_core": None,
        "job_rss_mib": None, "observed_peak_job_rss_mib": None,
        "ollama_cpu_percent_one_core": None, "ollama_rss_mib": None,
        "host_cpu_percent": None, "host_ram_used_mib": None,
        "host_ram_total_mib": None, "host_ram_percent": None,
        "process_count": 0, "sample_count": 0,
        "gpu_status": "not_instrumented",
        "note": "Live viewer-poll samples only; observed peaks reset on API restart. "
                "CPU 100% = one logical core. Process RSS can count shared memory twice."
    }
    with _LOCK:
        past = _SAMPLES.get(key)
        if past:
            result["observed_peak_job_rss_mib"] = past.get("peak_mib")
            result["sample_count"] = past.get("count", 0)
    if state not in _ACTIVE:
        result["monitor_state"] = "finished"
        return result
    if psutil is None:
        result["monitor_state"] = "unavailable"
        result["note"] = "Resource monitoring requires psutil in the API Python environment."
        return result
    launcher = directory / "launcher.json"
    if not launcher.is_file():
        return result
    launcher_info = _load_json(launcher)
    pid = launcher_info.get("pid")
    if type(pid) is not int or pid <= 0:
        result["monitor_state"] = "unavailable"
        result["note"] = "Worker launcher has no valid PID."
        return result
    launched = _time_seconds(launcher_info.get("at") or launcher_info.get("started_at"))
    if launched is None:
        launched = _time_seconds(status.get("created_at"))
    process = _read_processes(pid, launched)
    if process is None:
        result["monitor_state"] = "unavailable"
        result["note"] = "Worker process is unavailable or its PID has been recycled."
        return result
    cpu_times, rss_bytes, count = process
    excluded = {pid for pid, _ in cpu_times}
    ollama_times, ollama_bytes = _ollama_processes(excluded)
    now = time.monotonic()
    try:
        virtual = psutil.virtual_memory()
        host_cpu = psutil.cpu_times()
    except (OSError, ValueError):
        virtual = None
        host_cpu = None
    mib = 1024 * 1024
    rss_mib = round(rss_bytes / mib, 1)
    result.update({
        "monitor_state": "live", "job_rss_mib": rss_mib,
        "ollama_rss_mib": round(ollama_bytes / mib, 1) if ollama_times else None,
        "process_count": count,
        "host_ram_used_mib": round(virtual.used / mib, 1) if virtual else None,
        "host_ram_total_mib": round(virtual.total / mib, 1) if virtual else None,
        "host_ram_percent": round(virtual.percent, 1) if virtual else None,
    })
    with _LOCK:
        previous = _SAMPLES.get(key)
        if previous is not None:
            interval = now - previous["at"]
            if 0 < interval < 60:
                result["job_cpu_percent_one_core"] = _cpu_delta_percent(previous["job_cpu"], cpu_times, interval)
                result["ollama_cpu_percent_one_core"] = _cpu_delta_percent(previous["ollama_cpu"], ollama_times, interval)
                result["host_cpu_percent"] = _host_cpu(previous["host_cpu"], host_cpu)
        peak = max(rss_mib, (previous or {}).get("peak_mib", 0))
        samples = 1 + (previous or {}).get("count", 0)
        _SAMPLES[key] = {
            "at": now, "job_cpu": cpu_times, "ollama_cpu": ollama_times,
            "host_cpu": host_cpu, "peak_mib": peak, "count": samples,
        }
        # Strictly bounded state for the local development API process.
        if len(_SAMPLES) > _MAX_CACHED:
            oldest = next(iter(_SAMPLES))
            if oldest != key:
                _SAMPLES.pop(oldest, None)
        result["observed_peak_job_rss_mib"] = peak
        result["sample_count"] = samples
    return result
