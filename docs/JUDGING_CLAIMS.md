# NIMBO | Judging claims register

**Scope:** HUMAIN Hackathon, Challenge 2 — Deliver AI with Less Compute. The three judging criteria supplied by the team are (1) Task Performance, (2) Robustness & Adaptation, (3) Evaluation & Iteration. The five "What your solution may solve" items are solution directions, not known point weights.

## Main claim: same-model optimization

- **Claim:** ONNX Runtime FP32 executed the **same model and tokenizer** as PyTorch FP32 on 800 held-out IMDb examples, with three paired warm CPU measurement rounds. The renewed local run on 2026-10-31 reported **28.5501 ms → 17.7637 ms mean inference (37.781% lower)** and **91.75% labeled accuracy for both paths** (734/800 each), agreement 100%, p95 **36.6819 ms → 23.6119 ms**. ONNX passed the local quality, latency, p95, and repeatability gates. Variation between runs is expected; prior run measured 33.632% lower mean inference.
- **Evidence:** `app/nimbo_validate_v21.py`, `results/independent_validation_v21.jsonl`, `tools/judging_evidence.py`, regenerated `results/judging/LATEST_HELDOUT_EVIDENCE.md`.
- **Limit:** IMDb evaluation labels are *not third-party audited*, heldout is separate from the 200 development examples but not a novel external dataset. Warm CPU inference latency is **not** fewer FLOPs, electrical energy, monetary cost or full HTTP latency. `eligible_for_review` does not mean deployed. Outputs may vary by hardware and environment; validate actual saved run before citing new numbers.

## Other capabilities — bounded claims

| Area | Can demonstrate | Do not claim |
|---|---|---|
| Model routing | Paired local benchmark and adaptive fallback history | Guaranteed faster p95: historical adaptive p95 exceeded baseline |
| Context | Preliminary input-token and local timing measurements on four Qwen tasks | General verified semantic equivalence on unseen tasks |
| Image | Measured Fast/Standard GPU execution and quality uncertainty handling | Full prompt certified, energy/financial savings, or guaranteed success |
| Adoption | Local review → shadow → canary and rollback *code paths* | Production customer deployment |
| Evidence | Distinguishes live measurements, local labeled trials, and unverified states | Third-party certification |

## Judging criteria test plan

### Task Performance
Re-run the fixed heldout study and report prediction accuracy, mean, p95, agreement, per-round comparisons, and device/scope. Do **not** promote a single-request unlabelled comparison to accuracy validation.

### Robustness & Adaptation
Exercise at least: labels present; labels absent; heldout gate reject; unavailable backend/gateway; Fast image `UNCERTAIN` no false success; fallback only when justified; local rollout without production switch. Save results as dated test records. These additional integration tests are not yet certified in the source snapshot.

### Evaluation & Iteration
Keep `pytest` result and dated raw runs, model revision, partition digest, hardware, commands, version pins, Git commit SHA and clear negative results. No editing of historical raw results. Future reruns should append new evidence.

## Wording for live judges

> "Our main evaluation uses the same pretrained sentiment model and the same held-out inputs. ONNX Runtime preserved 91.75% measured IMDb reference-label accuracy and reduced warm CPU inference latency by about 37.8% in our latest three-round local run. We do not equate latency with measured energy or cloud-billing savings. NIMBO records the route, quality gate, latency, and audit boundary."
