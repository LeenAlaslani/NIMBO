# NIMBO — final judging evidence (local measured results)

**Model:** `lvwerra/distilbert-imdb` | **run:** `heldout_7830c2ff573a47ae`
**Artifact SHA-256:** `a4d615ada60cd7e9c4869939bf0039c42e717fdad77db0875279ceedaca39c6a`
**Baseline:** PyTorch FP32, 4 threads | **Candidate:** ONNX Runtime FP32, 2 threads

## Task Performance and compute

| Metric | Result | Scope |
|---|---:|---|
| Held-out labeled accuracy | 91.75% → 91.75% | 800 IMDb test-subset texts; 3 paired CPU rounds |
| Warm forward latency reduction | **14.79%** | Mean per-text, excludes setup and HTTP |
| Process CPU-time reduction | **2.54%** | Aggregate user+system CPU time across process threads, measured in one local run |

**Development choice:** 200 dev examples × 2 rounds; selected 2 ONNX threads before checking 800 held-out examples.

## Robustness and adoption

- Engine-level CPU concurrency: **6/6 PASS**, workers 1, 2, 4.
- Temporary localhost HTTP inference: **6/6 PASS**, workers 1, 2, 4. This is not the primary NIMBO UI/API.
- Isolated real adoption router over HTTP: **PASS**; Shadow 6 paired requests, Canary 5 paired safety checks and 2 candidate-only requests, followed by **verified rollback** to PyTorch.
- No customer production deployment occurred. Shadow calls both models and is not compute saving.

## Limitations and auditability

- The IMDb holdout is disjoint from this tuning set, but was previously evaluated in earlier runs; it is **not** a new independent external test.
- CPU time improved in this measurement, but the small positive margin needs additional runs for statistical confidence; no energy, FLOPs, dollar cost or individual-model RAM savings measured.
- Unlabeled HTTP/concurrency samples establish operation and agreement on that small test, **not** classification accuracy.
- Candidate approval means *eligible for manual review*, not automatically deployed.
- Source-file SHA-256 digests below identify local raw logs; they do not make local logs publicly available. Public summary is a derived evidence index.

## Local raw evidence index (hashes only)

| Source | Local relative file | SHA-256 |
|---|---|---|
| development | `results/resource_tuning_v3/latest_development_selection.json` | `a58343b19e5a2251eae76d53affebe83319c8a0dbd89ecc5a3a20df4b80b7373` |
| heldout_log | `results/independent_validation_v21.jsonl` | `924b5129e4d01d75a2d66ea998a2e124a4d5b69d7249dc469da59fdef8d314f3` |
| engine_stress | `results/concurrency_probe_v1/stress_heldout_7830c2ff573a47ae_20261031_080416.json` | `0b44c375960e406274622d6584c2a7055328c4f367254e14c468c2d9e0280a8f` |
| http_stress | `results/http_probe_v1/http_stress_heldout_7830c2ff573a47ae_20261031_080901.json` | `6dc9d30f11feeb18da48cbe404f9d0d4fb2391200a259b905eb899ebb9078585` |
| adoption | `results/adoption_proof_v1/session_20261031_081425_498392/report.json` | `a17efaf575a789c309b91b1aa2db3677c98a793dd6bc929e12406f4081bb4654` |

## How to reproduce

See [`CPU_THREAD_TUNING_STEP13.md`](CPU_THREAD_TUNING_STEP13.md), [`HTTP_PROBE_STEP16.md`](HTTP_PROBE_STEP16.md), and [`ADOPTION_PROOF_STEP17.md`](ADOPTION_PROOF_STEP17.md).
Do not rerun the locked held-out tuner merely to improve a number; a genuinely fresh external dataset is a separate next evaluation.
