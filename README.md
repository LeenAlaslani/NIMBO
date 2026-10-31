# NIMBO — Neural Intelligence for Multimodal Budget Optimization

**HUMAIN Hackathon · Challenge 2 — Deliver AI with Less Compute**

NIMBO is a local, quality-aware AI inference optimization lab. It compares baseline and candidate execution paths on identical inputs; records route choices, quality evidence, and CPU/GPU inference timing; and refuses to label unproven resource or task-quality claims as verified.

## Why NIMBO solves Challenge 2

**Business scenario:** A team classifies large volumes of customer feedback using an existing pretrained sentiment model. Replacing the model or making an unmeasured speed claim is not sufficient: NIMBO evaluates an execution-path optimization against a comparable baseline, checks labeled accuracy and runtime gates, and records when the candidate is eligible for review rather than silently deploying it. This is a **local demonstrator**, not an integrated production customer deployment.

**Optimization mechanism:** In the principal validated case, NIMBO executes the *same* fixed model and tokenizer using PyTorch FP32 and ONNX Runtime FP32. It compares the same IMDb evaluation inputs using alternating execution order and repeated runs. Lower warm CPU latency is a measured **runtime efficiency improvement**; this result alone is **not** a measurement of reduced arithmetic operations (FLOPs), electrical energy, or cloud billing.

**What judges can inspect:** Source and comparison gate in `app/nimbo_validate_v21.py`, regression tests in `tests/`, the summary at [`results/judging/LATEST_HELDOUT_EVIDENCE.md`](results/judging/LATEST_HELDOUT_EVIDENCE.md), the methods in [`docs/REPRODUCE_RESULTS.md`](docs/REPRODUCE_RESULTS.md), and the scope of every claim in [`docs/JUDGING_CLAIMS.md`](docs/JUDGING_CLAIMS.md).

<!-- NIMBO_FINAL_PROOF_BEGIN -->
## Primary proof: lower warm inference latency and process CPU time

With PyTorch FP32 (4 threads) versus ONNX Runtime FP32 (2 threads), the locked local 800-example evaluation measured **14.79% lower mean warm forward latency** and **2.54% lower process CPU time**, with **91.75% accuracy on both paths**. The candidate is *eligible for review*, not deployed. CPU-time savings are small and not an energy or billing measurement.

Full lineage and caveats: [`docs/FINAL_JUDGING_EVIDENCE.md`](docs/FINAL_JUDGING_EVIDENCE.md).
<!-- NIMBO_FINAL_PROOF_END -->

## Earlier latency-only comparison (not CPU-time evidence)

This earlier 37.78% speedup was measured under a different execution configuration. It is **latency-only evidence**, not evidence of reduced process CPU time. The primary dual-objective result above is the judged compute-efficiency claim.

On a fixed **800-example IMDb held-out** set, the same pretrained model was evaluated using **PyTorch FP32 versus ONNX Runtime FP32**, with three paired warm CPU rounds. In the renewed 2026-10-31 local run, recorded mean forward latency fell **28.5501 → 17.7637 ms** (**37.78% lower**), and labeled accuracy remained **91.75%** (734/800 per path). P95 improved from 36.6819 to 23.6119 ms; all local decision gates passed. The candidate is **eligible for human review, not production deployed**. Older runs measured different latency improvements; hardware and runtime effects matter.

**Important boundaries:** This is evidence of lower *warm CPU inference latency*, not electrical energy, FLOPs, cloud cost, or external quality certification. The evaluation is disjoint from a 200-example development subset of the same IMDb source; it is not an independent third-party assessment. Full HTTP and setup costs are not included in the quoted warm times.

## Architecture

- **CAPTAIN:** intake & requirements
- **TOKI:** context/token work when applicable
- **ROUTA:** routing and alternative comparisons
- **MAKO:** real model inference
- **VERA:** label-based guardrails and uncertainty
- **METRI:** measured performance and provenance

UI: React + React Three Fiber (`app/web/`). API: FastAPI (`app/mission_api.py`). Text: DistilBERT/BERT, Qwen, ONNX Runtime and experimental INT8 paths. Image: Stable Diffusion 1.5 with LCM-LoRA via optional real local GPU gateway and VERA. The VERA image pathway does **not** certify all attributes or scene relationships; `UNVERIFIED` is not a success or failure label.

## Reproduce

See [`docs/REPRODUCE_RESULTS.md`](docs/REPRODUCE_RESULTS.md) for tested commands and [`docs/JUDGING_CLAIMS.md`](docs/JUDGING_CLAIMS.md) for claim limits. After an actual local validation:

```bash
python -m pytest -q
python -m app.nimbo_validate_v21 --samples 800 --rounds 3 --no-export
python tools/judging_evidence.py
```

Read `results/judging/LATEST_HELDOUT_EVIDENCE.md` and keep dated raw logs. Some model dependencies and weights are not bundled; the GPU is a separate optional worker. The entire platform has **not** been independently certified or exhaustively browser-tested.

## Judging map

- **Task Performance:** label-aware held-out accuracy and latency gates.
- **Robustness & Adaptation:** multiple paths, conservative uncertainty handling, fail-closed decision gates, staged local rollout; outstanding cases documented.
- **Evaluation & Iteration:** unit tests, reproducible commands, model revision, dataset fingerprint, recorded measurements and explicit limits.

See [`docs/ROBUSTNESS_MATRIX.md`](docs/ROBUSTNESS_MATRIX.md) and [`docs/GITHUB_PREP.md`](docs/GITHUB_PREP.md). This repository is a local development/review system, not a production SaaS offering.

## Continuous integration and repository review

The [GitHub Actions checks](.github/workflows/ci.yml) run dependency-light Python regression tests and build the React frontend from a clean checkout. They **do not** download large model weights, run the local Mac GPU image worker, or reproduce the measured 800-example ONNX benchmark. For what is (and is not) automated, see [`docs/CI_AND_SUBMISSION.md`](docs/CI_AND_SUBMISSION.md).

A first-time reviewer can inspect the repository without GPU access; real measured inference must be reproduced separately on a compatible machine with model dependencies and downloads available.

<!-- NIMBO_CROSS_DOMAIN_RELEASE_BEGIN -->
## New-domain check — TweetEval (locked settings)

The pinned ONNX Runtime FP32 2-thread candidate retained **82.88% binary TweetEval accuracy** (PyTorch: **82.88%**; **100.00%** prediction agreement) on a precommitted 800-row positive/negative TEST subset. It measured **61.79% lower mean warm inference latency** and **52.72% lower process CPU time** in this local test. **Review-only; not deployed.** TweetEval accuracy is below IMDb's 91.75% due to domain/task differences; neutral class excluded. No electrical-energy or cloud-cost claim.

See [`docs/CROSS_DOMAIN_PUBLIC_EVIDENCE.md`](docs/CROSS_DOMAIN_PUBLIC_EVIDENCE.md) for the source digest, fixed protocol, limitations and audited fields. The full three-class TweetEval task was **not** evaluated.
<!-- NIMBO_CROSS_DOMAIN_RELEASE_END -->
