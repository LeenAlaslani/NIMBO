# NIMBO — held-out judging evidence (local run)

- **Run ID:** `heldout_cff50858e9a24db2`
- **Recorded UTC:** 2026-10-31T06:06:14.449768+00:00
- **Model:** `lvwerra/distilbert-imdb`
- **Model revision:** `0fc02cd68445b599a9cb2da2368050e7fb31d29a`
**Held-out digest:** `10fd80603eaad484728ac6c00c2c702932b903c9f11c08ebfdc175b7e914ac68`

| Metric | PyTorch FP32 | ONNX FP32 |
|---|---:|---:|
| Warm mean inference | 28.5501 ms | 17.7637 ms |
| Warm p95 inference | 36.6819 ms | 23.6119 ms |
| Reference-label accuracy | 91.75% | 91.75% |

- **Mean warm CPU inference reduction:** 37.78%
- **Tokenization + forward local reduction:** 37.16%
- **Prediction agreement:** 100.00%
- **Paired test:** 800 disjoint IMDb records, 3 rounds, 2400 measured passes per path.
- **Gate decision:** `eligible_for_review`; not a production deployment.

## Robustness / repeatability across rounds

| Round | Baseline mean (ms) | ONNX mean (ms) |
|---:|---:|---:|
| 1 | 27.4698 | 18.0707 |
| 2 | 29.3646 | 17.6885 |
| 3 | 28.8160 | 17.5320 |

## Trust boundary

Reference labels are the local IMDb evaluation labels on a disjoint holdout, **not third-party audited quality**.
Measurements are **warm CPU inference latency**, not reduced FLOPs, electrical energy or dollar costs.
Setup/model loading and HTTP/browser time are **not** part of the warm inference numbers.
Passing the gate means **eligible for manual review**, not deployed or production certified.

Environment snapshot: `{"machine": "arm64", "processor": "arm", "system": "Darwin"}`; Python 3.11.9, torch 2.15.0, onnxruntime 1.31.1.
