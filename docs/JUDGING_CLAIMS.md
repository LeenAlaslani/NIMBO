# NIMBO | Judging claims register

**Scope:** HUMAIN Hackathon, Challenge 2 — Deliver AI with Less Compute. The three judging criteria supplied by the team are (1) Task Performance, (2) Robustness & Adaptation, (3) Evaluation & Iteration. The five "What your solution may solve" items are solution directions, not known point weights.

## Primary dual-objective claim: faster warm inference with lower measured process CPU time

- **Measured result:** A development-only CPU-thread sweep chose ONNX Runtime FP32 with **2 threads**, compared with PyTorch FP32 using **4 threads** on the same pinned IMDb model. The subsequent 800-example, 3-round local heldout evaluation reported **14.79% lower mean warm forward latency**, **2.54% lower process CPU time**, and **91.75% classification accuracy in both paths**.
- **Evidence:** [`FINAL_JUDGING_EVIDENCE.md`](FINAL_JUDGING_EVIDENCE.md), [`FINAL_JUDGING_EVIDENCE.json`](FINAL_JUDGING_EVIDENCE.json), [`CPU_THREAD_TUNING_STEP13.md`](CPU_THREAD_TUNING_STEP13.md), `app/nimbo_validate_v21.py`, `tests/` and `tools/`. Local raw logs are gitignored; GitHub checks only public summary consistency and unit tests.
- **Limit:** The 2.54% CPU-time decrease comes from one local run with no confidence interval, and the heldout set had been used in earlier evaluations. No measured energy, FLOPs, cloud billing or per-path peak RAM reduction. Candidate was eligible for **manual review**, not production deployed. Temporary HTTP and isolated Shadow/Canary/Rollback checks are not proof of production service behavior.

## Earlier latency-only proof (not lower CPU time)

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
Exercise at least: labels present; labels absent; heldout gate reject; unavailable backend/gateway; Fast image `UNCERTAIN` no false success; fallback only when justified; local rollout without production switch. Save results as dated test records. Local HTTP and isolated adoption tests are documented in [`FINAL_JUDGING_EVIDENCE.md`](FINAL_JUDGING_EVIDENCE.md); production and primary-UI integration remain unverified.

### Evaluation & Iteration
Keep `pytest` result and dated raw runs, model revision, partition digest, hardware, commands, version pins, Git commit SHA and clear negative results. No editing of historical raw results. Future reruns should append new evidence.

## Wording for live judges

> "NIMBO tested execution configurations with quality and resource gates. Its development-selected ONNX setting reduced warm inference latency by 14.79% and measured process CPU time by 2.54% on the local 800-example three-round evaluation, while maintaining 91.75% measured accuracy. This is not energy or cloud-cost proof, and the small CPU margin needs replication. We also tested isolated HTTP concurrency, Canary and Rollback without deploying to customers."

<!-- NIMBO_CROSS_DOMAIN_RELEASE_BEGIN -->
## New-domain evaluation (TweetEval binary subset)

- **Result:** 82.88% → 82.88% accuracy; 100.00% agreement; 61.79% lower warm latency; 52.72% lower process CPU time, locked PyTorch 4 threads versus ONNX 2 threads.
- **Evidence:** [`CROSS_DOMAIN_PUBLIC_EVIDENCE.md`](CROSS_DOMAIN_PUBLIC_EVIDENCE.md) and the pinned source code in `tools/external_domain_v1.py`.
- **Limit:** Binary-only TweetEval TEST subset; neutral excluded, pretrained model exposures unknown, local one-machine evidence, not deployed, no energy/FLOPs/billing proof.
<!-- NIMBO_CROSS_DOMAIN_RELEASE_END -->
