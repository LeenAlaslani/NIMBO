# Reproduce NIMBO's main judging proof

**Workspace:** macOS local CPU, NIMBO source root. Preserve existing `.venv`, and do not publish credentials or private datasets.

1. `cd ~/Desktop/NIMBO && source .venv/bin/activate`
2. `python --version && node --version && npm --version`
3. `python -m pytest -q`
4. `python -m app.nimbo_validate_v21 --samples 800 --rounds 3 --no-export | tee "results/heldout_judging_$(date +%Y%m%d_%H%M%S).txt"`
5. `python tools/judging_evidence.py`
6. Read `results/judging/LATEST_HELDOUT_EVIDENCE.md`. Verify decision is actually `eligible_for_review` before presenting a pass. Previous results must not be replaced with fabricated data.

**Environment:** first-run downloads/model graph export can require Internet, package dependencies, and disk space. Model setup/warmup and full HTTP/browser latency are excluded from quoted warm inference numbers. Same model revision, fixed heldout indices, alternating execution order and 3 rounds; local result varies across runs.

## Web demo (separate terminals)

Terminal 1: `cd ~/Desktop/NIMBO && source .venv/bin/activate && python -m uvicorn app.mission_api:app --reload --port 8000`

Terminal 2: `cd ~/Desktop/NIMBO/app/web && npm ci && npm run dev`

Navigate to `http://127.0.0.1:5173`. Select **Customer feedback classification → Validate ONNX · 800 held-out → Enter Mission Floor**. Run the local validation with a real backend. This is distinct from a **Single request** comparison. Without reference labels, a single request legitimately displays "Accuracy not evaluated".

Optional GPU image worker, only if the dependencies and local MPS/CUDA environment are available:
`cd ~/Desktop/NIMBO && source .venv/bin/activate && python -m uvicorn app.nimbo_image_worker_v22:app --port 8011`. GPU execution needs a separate real health/inference check. Images are an experimental secondary proof.

## What still needs a manual demonstration

- Fresh browser walkthrough Workspace → Mission Floor → METRI receipt → Evidence Center, with screenshots and a full run ID.
- Test backend/GPU disconnection and verify the UI does not claim health based on configuration alone.
- Add cross-domain heldout task performance and repeated resource instrumentation before claiming broad robustness or compute/energy reductions.
- Check the GitHub commit used for judging exactly matches the source tested on the Mac.
