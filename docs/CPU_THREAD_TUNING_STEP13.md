# NIMBO CPU Thread Tuning — Step 13

## Why

The October 31 Step 11 run had **31.41% lower mean warm CPU inference wall time** at equal measured IMDb classification accuracy (91.75%), but **18.72% more process CPU time**. A latency advantage is **not** a CPU-work saving. The Step 12 `cpu_time` objective correctly rejects that result.

## Development-only search, then locked evaluation

The dev tuner uses a pinned, SHA-256-verified previously exported ONNX graph and exactly **200 IMDb development examples** from the fixed seeded test partition (`ids[0:200]`). The baseline always uses PyTorch with 4 intra-op threads; candidate ONNX sessions are **1, 2, and 4 threads**, constructed one at a time. Each comparison uses two paired CPU rounds with alternating execution order. It measures warm pretokenized forward wall time and process CPU time and also checks labels/prediction agreement. Model download, ONNX session creation, tokenization, and warmup are excluded from this sweep's forward timings. There is no billing, joule, or FLOP claim.

Each candidate must have: dev accuracy drop at most 1 percentage point; agreement >=99.5%; >=5% mean wall-latency improvement; >=2% mean process CPU reduction (screening margin, **not** a statistical confidence bound); no p95 wall regression; and both lower wall and lower CPU in **each** development round. If multiple pass, choose the largest CPU reduction. If none passes, report `NO_ELIGIBLE_CANDIDATE`, keep the baseline, and **do not** run the heldout step.

The final locked evaluation (only if a candidate passes development) uses **800 distinct held-out examples**, 3 paired rounds, and the existing `cpu_time` quality/latency/p95/CPU gates. It can stage the artifact *for review only*; it does not deploy or overwrite any production endpoint.

**Important limitation:** The holdout was already evaluated in prior NIMBO trials. It is disjoint from the 200 tuning inputs, but not a previously untouched or externally audited test set. Do not present it as virgin unseen data. Process CPU time includes the Python process's threads and is sensitive to background/system contention; memory is whole-process peak, not per-path. Actual results must come from your Mac.

## Install and execute

```bash
cd ~/Desktop/NIMBO_GITHUB_READY_20261031
~/Desktop/NIMBO/.venv/bin/python -m pytest -q
~/Desktop/NIMBO/.venv/bin/python -m tools.cpu_thread_tuner_v3 --phase development
```

Look for `Decision: SELECTED_DEV_ONLY` **or** `Decision: NO_ELIGIBLE_CANDIDATE`. Send the console summary and `results/resource_tuning_v3/latest_development_selection.json` for review. Do not run heldout evaluation until the development outcome is reviewed.

If a candidate passes and its policy is approved for evaluation, the following performs the heldout check with a locked, once-only selection:

```bash
~/Desktop/NIMBO/.venv/bin/python -m tools.cpu_thread_tuner_v3 --phase heldout
```

This writes a new immutable append-only local heldout record under `results/independent_validation_v21.jsonl`; the selection is marked used before the run and cannot be reused for iterative heldout tuning. Keep saved outputs as evidence, not as claims of energy or dollars saved.
