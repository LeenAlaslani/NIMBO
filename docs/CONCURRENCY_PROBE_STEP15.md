# NIMBO Step 15 — Local inference-engine concurrency proof

**Scope:** Read-only CPU model inference. It does not call the running FastAPI service, change NIMBO's adoption state, deploy a candidate, or download external data. It uses the **last eligible heldout artifact** and requires an already locked selection from development that passed its CPU objective in heldout.

Run from the isolated worktree; existing NIMBO background terminals may remain open. Avoid GPU tasks **only for a clean benchmark**; concurrency capacity is tested deliberately inside this probe. It is normal for CPU-based tests to share the Mac with other services; background activity may affect the results.

```bash
cd ~/Desktop/NIMBO_GITHUB_READY_20261031
~/Desktop/NIMBO/.venv/bin/python -m pytest -q
~/Desktop/NIMBO/.venv/bin/python -m tools.concurrency_probe_v1 --requests 24 --repeats 2
```

The tool tests concurrent requests at 1, 2 and 4 workers, twice each, alternating the baseline and candidate order. It runs matched **unlabeled synthetic classification texts**, not the 800-label heldout set. The two execution paths share one pinned model and tokenizer; the PyTorch and ONNX models stay in the same process. It checks for failures and prediction parity and reports each path's throughput and p95 burst-response latency. A worker exception or parity mismatch yields `FAIL` and exits nonzero.

**Interpretation:** This is a CPU **engine-level burst stress probe**, not HTTP/network/production testing. Queue wait inside the executor is included in burst-response latency. Model loading and warmup are excluded. CPU process time covers all process threads and cannot isolate background threads. Neither memory footprint per model nor energy is measured. Do **not** infer model accuracy from unlabeled inputs, and do **not** automatically approve or adopt a candidate based on this probe. No changes to adoption/canary/rollback state.

Next: inspect the JSON in `results/concurrency_probe_v1/`, rerun under well-characterized load if needed, and separately test the FastAPI service with client timeouts/reliability gates. Data selection and thresholds must remain fixed before any fresh external holdout validation.
