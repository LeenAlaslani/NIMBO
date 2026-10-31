# Reproducing NIMBO judging evidence

## Primary dual-objective result: CPU work plus warm latency

The selected setting is **PyTorch FP32, 4 threads** versus **ONNX Runtime FP32, 2 threads**, chosen on 200 development examples and evaluated on 800 separate IMDb examples in three rounds. See [`CPU_THREAD_TUNING_STEP13.md`](CPU_THREAD_TUNING_STEP13.md) for the actual development/locked-heldout commands and stopping rules. Do **not** rerun the already-consumed locked selection just to improve a result. The public summary is in [`FINAL_JUDGING_EVIDENCE.md`](FINAL_JUDGING_EVIDENCE.md).

From a source checkout, verify the **public summary's internal consistency only** (not the absent private raw logs):

```bash
python -m tools.verify_public_proof_v1
python -m pytest -q
```

The benchmark itself requires the pinned pretrained model, IMDb dataset access, appropriate dependencies and a Mac CPU environment; a fresh external holdout is still required for stronger generalization claims. The current published summary records one local dual-objective result, not a new remote benchmark run.

## Earlier latency-only reproduction (does not test CPU-time savings)

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

<!-- NIMBO_CROSS_DOMAIN_RELEASE_BEGIN -->
## Cross-domain fixed TEST check

The 800-row TweetEval binary test was run once after locking ONNX CPU threads. It must not be used to retune the candidate.
Source code: `tools/external_domain_v1.py`; protocol: [`EXTERNAL_DOMAIN_STEP22.md`](EXTERNAL_DOMAIN_STEP22.md); public aggregate evidence: [`CROSS_DOMAIN_PUBLIC_EVIDENCE.md`](CROSS_DOMAIN_PUBLIC_EVIDENCE.md).
The local raw report at `results/external_domain_v1/tweet_eval_binary_test_800.json` is intentionally excluded from Git.
Use `python -m tools.verify_cross_domain_public_v1` to check the published summary without re-running models or downloading datasets.
<!-- NIMBO_CROSS_DOMAIN_RELEASE_END -->
