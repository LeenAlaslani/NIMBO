# Step 12 — Objective-aware review (do not equate latency with compute)

In the real 31 October 2026 run, PyTorch warm CPU forward was 24.4127 ms,
ONNX was 16.7448 ms (31.41% lower wall time), while process CPU time was
51.8836 vs 61.5972 ms (18.72% **more** accumulated CPU time). Accuracy
remained 734/800 (91.75%) and predictions agreed 100%.

This is a mixed outcome. The CPU process time includes time across threads;
it is not power/energy. The measured peak RSS is shared for both models,
so does not establish per-variant RAM savings.

**New policy:** `optimization_objective=latency` retains the existing
latency review behavior but explicitly labels CPU regression and forbids
a less-CPU claim. `optimization_objective=cpu_time` requires the quality,
latency, P95, round-repeatability **and** CPU-time non-regression gates.
Missing CPU time fails closed. Neither policy approves production deployment.

Reassess the historical Step 11 run *without rerunning models*:

```bash
cd ~/Desktop/NIMBO_GITHUB_READY_20261031
~/Desktop/NIMBO/.venv/bin/python -m tools.resource_objective_v2
```

The expected historical result is `Latency review: PASS`,
`CPU-time review: FAIL`, `Recommendation: LATENCY_ONLY`. These are
derived interpretations of recorded measurements, **not a new validation**.

To run a **future** 800x3 heldout experiment with explicit CPU budget:

```bash
~/Desktop/NIMBO/.venv/bin/python -m app.nimbo_validate_v21 \
  --samples 800 --rounds 3 --objective cpu_time --max-cpu-increase 0
```

A candidate that breaks the CPU-time budget will be rejected, and **no
staged ONNX artifact will be created for that new run**. Historic staged
review candidates remain untouched and never deploy automatically.

Do not optimize thread count on the same final heldout test partition:
use separate calibration/development inputs; freeze the setup, then run
genuinely unused final evaluation examples. To prove performance under
concurrent requests, measure throughput, P95, quality, CPU budget and
queueing with a controlled load test (not demonstrated here).

Direct-file execution is also supported after Step12A; when running from repository root, prefer `python -m tools.resource_objective_v2`.
