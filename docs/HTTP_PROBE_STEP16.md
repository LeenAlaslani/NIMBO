# Step 16 — Local HTTP concurrency proof (isolated; read-only)

This is an **actual HTTP request/load probe**, not an in-process engine-only probe. It starts an isolated FastAPI service **on a temporary local 127.0.0.1 port**, uses the pinned exported ONNX candidate and the identical PyTorch model from the verified Step 14 record, sends matched synthetic classification requests, measures client-observed response latency and throughput, then automatically stops the temporary service.

**Important limitation:** This is a purpose-built minimal HTTP wrapper around the actual model engines; it is **not** the existing NIMBO Mission API or its adoption/canary endpoint. It does **not** mutate local adoption state, deploy models, or prove production readiness. Its synthetic text has no reference labels, so it checks model agreement, not task accuracy. It runs both variants sequentially per condition and alternates the order over repeats. Model loading/warmup excluded.

From the separate `NIMBO_GITHUB_READY_20261031` folder:

```bash
cd ~/Desktop/NIMBO_GITHUB_READY_20261031
~/Desktop/NIMBO/.venv/bin/python -m pytest -q
~/Desktop/NIMBO/.venv/bin/python -m tools.http_probe_v1 --requests 24 --repeats 2
```

The probe validates the development-only thread selection + held-out acceptance, staged manifest, SHA-256 and model revision. It uses only the loopback interface, rejects browser Origins, restricts payload size, and forces local cached model weights. If the model weights are not locally available, it fails rather than claiming a benchmark. It never uses port 8000, 8011, or 5173. The server subprocess is stopped even on an error; see `results/http_probe_v1/server_*.log` for startup failures.

Outputs under `results/http_probe_v1/` are ignored by Git unless selected public metadata are deliberately exported. `PASS` means every request succeeded and both engines agreed; *performance flags* remain descriptive. A clean outcome is not an energy, billing, FLOPs, customer-accuracy or scaled-production-load claim.

Follow-up: perform an isolated state-machine test of the **actual adoption endpoints**, exercising shadow/canary/rollback (with explicit opt-in), and a fresh, unseen held-out domain evaluation. Do not choose thread settings based on the Step 16 traffic measurements.
