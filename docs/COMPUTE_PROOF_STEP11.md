# Step 11 — Resource and staged ONNX proof (local macOS)

## Why this matters

Previously NIMBO measured **latency**, and exported an ONNX review candidate only
when `--no-export` was omitted. The new optional resource evidence records
`time.process_time()` for each paired inference (including CPU time across threads),
as well as **peak RSS for the shared benchmark process**. These are different from
energy, FLOPs, money and **per-model** peak memory.

## Exact reproducible commands (run from isolated `~/Desktop/NIMBO_GITHUB_READY_20261031`)

```bash
cd ~/Desktop/NIMBO_GITHUB_READY_20261031
~/Desktop/NIMBO/.venv/bin/python -m pytest -q
~/Desktop/NIMBO/.venv/bin/python -m app.nimbo_validate_v21 --samples 800 --rounds 3 | tee "results/compute_proof_run_$(date +%Y%m%d_%H%M%S).txt"
~/Desktop/NIMBO/.venv/bin/python tools/compute_proof_v1.py
~/Desktop/NIMBO/.venv/bin/python tools/staged_candidate_smoke_v1.py
```

The validation will **only stage** `results/onnx_review_candidates_v21/<run_id>/`
with `model.onnx`, a locally saved tokenizer, and a SHA-256 manifest if the quality,
latency, p95 and repeatability gates **actually pass**. It will not deploy,
auto-switch routes, or activate shadow/canary. Keep at least **1 GB free disk**
for staging, cached weights and temporary artifacts; actual requirements vary. The isolated worktree has its OWN `results/` directory; nothing is copied into or deployed from the live NIMBO app.

The smoke tool uses the existing adoption verifier to check saved heldout record,
manifest, exact SHA-256 and model revision before running eight real paired probes
with the staged ONNX and original PyTorch model. If a probe mismatches or drifts
more than 2.5 percentage points it raises a failure rather than claiming a pass.
These eight texts are **diagnostic only**; they are not a new held-out accuracy test
or proof that broad-domain quality is preserved.

If the validation gate rejects, the missing staged candidate is expected; STOP,
report the actual failure, and do not run smoke as if an export succeeded.

New raw resource reports go into `results/compute_proof_v1/` and are **gitignored**.
Keep them private until their accuracy and provenance have been reviewed.

### Limitations

- CPU process time includes Python and CPU threads from PyTorch and ONNX and may
  include concurrent activity in the benchmarking process. It is **not** energy.
- Measured process RSS includes **both loaded models**, tokenizer, interpreter,
  and earlier allocations. We cannot infer relative model RAM consumption.
- CPU measurements require a **fresh run**; never infer them from historical
  `37.78%` warm wall-time improvement.
- The validation is on an IMDb heldout subset disjoint from NIMBO's local dev
  samples, not a third-party or cross-domain quality certification.
- Shadow operation deliberately runs both models: it does **not** save compute.
- A staged review candidate is **not deployed**. Full service adoption is a
  separate explicit opt-in trial with safety sampling and rollback.

### Failure handling

If the speedup or accuracy gate fails, **do not edit policy thresholds to force approval**. Keep the rejected record and investigate thermal/load and model differences. Stop before the staged smoke command; a rejection is a meaningful robustness result. Do not commit local ONNX binaries or raw datasets to the public repo.
