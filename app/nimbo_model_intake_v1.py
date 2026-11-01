"""NIMBO Auto-Optimizer V1, milestone 1: real safe-ish public-model baseline.

This module does NOT optimize, quantize, publish, or deploy a model. It loads a
pinned public Hugging Face sequence classifier and measures actual CPU inference.
Only trusted built-in Transformer architectures and safetensors are supported.
Remote model execution is not a general-purpose security sandbox; use a restricted
worker for public uploads before offering the service on the Internet.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_DATASET_BYTES = 1024 * 1024
MAX_MODEL_WEIGHT_BYTES = 750 * 1024 * 1024
MAX_EXAMPLES = 100
MAX_TEXT_CHARS = 2000
MAX_TOKEN_LENGTH = 256
ALLOWED_MODEL_TYPES = frozenset({"bert", "distilbert", "roberta", "albert", "electra"})
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
WEIGHT_NAME_RE = re.compile(r"(?:^|/)(?:model\.safetensors|model-\d{5}-of-\d{5}\.safetensors)$")


@dataclass(frozen=True)
class ModelIntake:
    model_id: str
    resolved_revision: str
    architecture: str
    num_labels: int
    id2label: dict[int, str]
    safe_weight_bytes: int
    weight_files: tuple[str, ...]
    license: str | None


def check_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id):
        raise ValueError("Model ID must be a public Hugging Face owner/repository ID, not a URL or file path")
    if any(part in {".", ".."} for part in model_id.split("/")):
        raise ValueError("Invalid model ID")
    return model_id


def inspect_hub_metadata(model_id: str, info, config, *, max_weight_bytes: int = MAX_MODEL_WEIGHT_BYTES) -> ModelIntake:
    """Validate already-fetched metadata and config; independently unit-testable."""
    check_model_id(model_id)
    if getattr(info, "private", False):
        raise ValueError("Private repositories are not accepted by this public-model intake")
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not REVISION_RE.fullmatch(sha):
        raise ValueError("Hub revision is not an immutable 40-character commit SHA")
    architecture = str(getattr(config, "model_type", ""))
    if architecture not in ALLOWED_MODEL_TYPES:
        raise ValueError(f"Unsupported architecture: {architecture!r}; supported: {', '.join(sorted(ALLOWED_MODEL_TYPES))}")
    num_labels = getattr(config, "num_labels", None)
    if isinstance(num_labels, bool) or not isinstance(num_labels, int) or not 2 <= num_labels <= 20:
        raise ValueError("Sequence classifier must declare 2–20 label classes")
    names = getattr(config, "id2label", None)
    if not isinstance(names, dict) or len(names) != num_labels:
        raise ValueError("Missing or inconsistent label definitions in model config")
    labels = {int(k): str(v) for k, v in names.items()}
    if set(labels) != set(range(num_labels)) or any(not x.strip() for x in labels.values()):
        raise ValueError("Invalid class-to-label mapping")
    weights = []
    total = 0
    for file in getattr(info, "siblings", []) or []:
        name = getattr(file, "rfilename", "")
        if not WEIGHT_NAME_RE.fullmatch(str(name)):
            continue
        size = getattr(file, "size", None)
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("Hub did not provide a verifiable safetensors weight size; refusing download")
        total += size
        weights.append(str(name))
        if total > max_weight_bytes:
            raise ValueError(f"Model weights exceed {max_weight_bytes} byte safety cap")
    if not weights:
        raise ValueError("No supported safetensors classifier weights; .bin/pickle weights are not accepted")
    raw_license = getattr(info, "cardData", None)
    if isinstance(raw_license, dict):
        license_name = raw_license.get("license")
    else:
        license_name = getattr(raw_license, "license", None)
    return ModelIntake(model_id, sha, architecture, num_labels, labels, total, tuple(weights), str(license_name) if license_name else None)


def inspect_public_classifier(model_id: str, *, max_weight_bytes: int = MAX_MODEL_WEIGHT_BYTES) -> ModelIntake:
    """Fetch public metadata, pin an immutable revision, then validate config."""
    check_model_id(model_id)
    from huggingface_hub import HfApi
    from transformers import AutoConfig
    info = HfApi().model_info(model_id, revision="main", files_metadata=True)
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not REVISION_RE.fullmatch(sha):
        raise ValueError("Hugging Face did not return a pinned revision")
    config = AutoConfig.from_pretrained(model_id, revision=sha, trust_remote_code=False)
    return inspect_hub_metadata(model_id, info, config, max_weight_bytes=max_weight_bytes)


def read_evaluation_file(path: str | Path) -> tuple[list[dict[str, str | None]], str]:
    """Read a bounded CSV/JSONL evaluation dataset without writing raw texts to evidence."""
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.lower() not in (".csv", ".jsonl"):
        raise ValueError("Evaluation dataset must be CSV or JSONL")
    raw = path.read_bytes() if path.stat().st_size <= MAX_DATASET_BYTES else None
    if not raw or len(raw) > MAX_DATASET_BYTES:
        raise ValueError("Evaluation dataset must be nonempty and at most 1 MB")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Evaluation dataset must be UTF-8") from exc
    if path.suffix.lower() == ".csv":
        from io import StringIO
        reader = csv.DictReader(StringIO(content))
        if not reader.fieldnames or "text" not in reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV must contain one text column; duplicate column names are not supported")
        rows = list(reader)
    else:
        rows = []
        for n, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {n}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {n} must be a JSON object")
            rows.append(obj)
    if not 2 <= len(rows) <= MAX_EXAMPLES:
        raise ValueError(f"Evaluation dataset must have 2–{MAX_EXAMPLES} rows")
    cleaned = []
    for i, row in enumerate(rows, 1):
        text = row.get("text")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_TEXT_CHARS:
            raise ValueError(f"Row {i}: text must be 1–{MAX_TEXT_CHARS} characters")
        label = row.get("label", row.get("expected_label"))
        if label is not None and not isinstance(label, (str, int)):
            raise ValueError(f"Row {i}: invalid label type")
        cleaned.append({"text": text.strip(), "label": str(label).strip() if label is not None and str(label).strip() else None})
    return cleaned, digest


def resolve_labels(rows: list[dict], mapping: dict[int, str]) -> list[int | None]:
    """Never guess mappings like LABEL_0=NEGATIVE; match config or numeric IDs."""
    aliases = {name.casefold(): idx for idx, name in mapping.items()}
    if len(aliases) != len(mapping):
        raise ValueError("Model class labels are ambiguous")
    result: list[int | None] = []
    for i, row in enumerate(rows, 1):
        raw = row.get("label")
        if raw is None:
            result.append(None)
        elif raw in {str(idx) for idx in mapping}:
            result.append(int(raw))
        elif str(raw).casefold() in aliases:
            result.append(aliases[str(raw).casefold()])
        else:
            raise ValueError(f"Row {i}: label {raw!r} does not match model label names/IDs: {mapping}")
    return result


def percentile(values: list[float], q: float) -> float:
    if not values or not 0 <= q <= 1:
        raise ValueError("Invalid percentile")
    vals = sorted(values)
    position = (len(vals) - 1) * q
    a = math.floor(position)
    b = math.ceil(position)
    return vals[a] * (b - position) + vals[b] * (position - a) if a != b else vals[a]


def summarize_baseline(predictions: list[int], expected: list[int | None], wall_ms: list[float], cpu_ms: list[float]) -> dict:
    n = len(predictions)
    if n < 2 or len(expected) != n or len(wall_ms) != n or len(cpu_ms) != n:
        raise ValueError("Invalid or incomplete inference measurements")
    if any(x < 0 or not math.isfinite(x) for x in wall_ms + cpu_ms):
        raise ValueError("Invalid timings")
    fully_labeled = all(label is not None for label in expected)
    return {"samples": n, "accuracy": round(sum(int(a == b) for a, b in zip(predictions, expected)) / n, 6) if fully_labeled else None,
            "accuracy_scope": "all_supplied_labels" if fully_labeled else "unverified_missing_labels",
            "mean_warm_forward_ms": round(statistics.mean(wall_ms), 4),
            "p95_warm_forward_ms": round(percentile(wall_ms, 0.95), 4),
            "process_cpu_forward_total_ms": round(sum(cpu_ms), 4),
            "note": "Per-item warm CPU model forward only; tokenizer, setup, downloads, memory, energy and billing excluded."}


def run_real_baseline(model_id: str, dataset_path: str | Path, *, cpu_threads: int = 2, max_length: int = 256, output_dir: str | Path = "results/auto_optimizer_baselines") -> tuple[Path, dict]:
    """Execute REAL pinned CPU inference and save an aggregate-only baseline report."""
    if not 1 <= cpu_threads <= 8 or not 32 <= max_length <= MAX_TOKEN_LENGTH:
        raise ValueError("CPU threads must be 1–8 and max_length must be 32–256")
    rows, digest = read_evaluation_file(dataset_path)
    spec = inspect_public_classifier(model_id)
    expected = resolve_labels(rows, spec.id2label)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.resolved_revision, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.model_id, revision=spec.resolved_revision, trust_remote_code=False,
        use_safetensors=True, low_cpu_mem_usage=False).cpu().eval()
    def forward(text: str):
        prepared = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
        start_wall = time.perf_counter_ns()
        start_cpu = time.process_time_ns()
        with torch.inference_mode():
            logits = model(**prepared).logits
        cpu = (time.process_time_ns() - start_cpu) / 1e6
        wall = (time.perf_counter_ns() - start_wall) / 1e6
        return int(logits.argmax(dim=-1).item()), wall, cpu
    for row in rows[:min(2, len(rows))]:
        forward(row["text"])
    preds, walls, cpus = [], [], []
    for row in rows:
        pred, wall, cpu = forward(row["text"])
        preds.append(pred)
        walls.append(wall)
        cpus.append(cpu)
    metrics = summarize_baseline(preds, expected, walls, cpus)
    report = {"schema_version": "nimbo_baseline_v1", "status": "BASELINE_ONLY_NOT_OPTIMIZED",
              "source": "public_huggingface_model", "model": asdict(spec), "dataset_sha256": digest,
              "metrics": metrics, "cpu_threads": cpu_threads, "max_length": max_length,
              "model_package_bytes": None,
              "limits": ["No optimized candidate, export or deployment has been generated.",
                         "Hub safetensors file sizes are source metadata, not a measured installed package size.",
                         "Public model code is not a security sandbox; restrict execution before remote multi-user deployment."]}
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = hashlib.sha256((spec.model_id + spec.resolved_revision + digest + str(time.time_ns())).encode()).hexdigest()[:16]
    path = out_dir / f"baseline_{run_id}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path, report
