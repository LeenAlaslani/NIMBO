# NIMBO CURRENT STATE — Auto-Optimizer V1 Milestone 1

- Updated (Asia/Riyadh): 2026-11-01 21:50:02 +0300
- Development branch: `develop/auto-optimizer-v1`
- GitHub baseline source: `main` at `710f4f152f00983db4e822df466fa1ea3325d5ba` (historically verified 2026-11-01)
- Feature commit SHA (created and verified before this checkpoint): `9b5b69f1cd1f22a531859d8b5fdfcf7cfa4bd705`
- Checkpoint commit SHA: use `git rev-parse HEAD` (a commit cannot contain its own SHA).
- GitHub synchronization: local code committed; remote branch status must be checked separately.
- GitHub CI status for this branch: NOT VERIFIED until a remote workflow has run.
- Local Mac workspace: `~/Desktop/NIMBO_AUTO_OPTIMIZER_V1` (installer default)
- Original running workspace: unchanged by this installer.

## Product decisions

NIMBO's primary mission remains: **Import supported user model → baseline → optimize → verify quality/compute → select/reject → downloadable runnable artifact**. The 3D workers, routing and comparisons support this mission.

## Verified milestone 1

- Added `app/nimbo_model_intake_v1.py`: public Hugging Face sequence-classifier intake; pins resolved model SHA; supports only built-in BERT/DistilBERT/RoBERTa/ALBERT/ELECTRA configs with `trust_remote_code=False` and safetensors weight size cap (750 MiB). No arbitrary Python code/pickle loading.
- Added `tools/run_nimbo_baseline_v1.py`: local CLI for **real CPU** baseline inference on 2–100 CSV/JSONL inputs. Measures warm forward latency, p95, process CPU time, reference-label accuracy if fully labeled. Writes aggregate JSON (no raw texts) in ignored `results/`.
- Added `tests/test_nimbo_model_intake_v1.py`: dependency-light tests for input acceptance, fail-closed metadata, labels and summaries.
- Added no endpoint, React UI, worker, candidate optimization, model exporter or cloud deployment in this milestone. Existing code untouched.
- Existing hardcoded ONNX/INT8 engine remains unchanged; its proven evidence remains valid only for its original pinned case.

## Tests and evidence

Full `python -m pytest -q`: PASS on local Mac (see terminal transcript).
- `npm ci` + `npm run build`: PASS on local Mac (see terminal transcript).
- Live external model loading and GPU: NOT RUN by installer. Must execute the documented baseline CLI on the Mac and report actual measured result.
- Benchmark versus optimized candidate: NOT RUN; not yet implemented for arbitrary imported models.
- Security: size cap and disabled remote custom code are basic controls, **not** OS-level isolation; do not expose unrestricted web uploads yet.

## Outstanding risks

- Build a restricted model worker, resource/time limits, tokenizer/model family checks, and license policy before a public multi-tenant service.
- Generalize existing ONNX and INT8 code to imported model specifications, not hardcoded model IDs.
- Independently reload candidate artifacts and verify quality before allowing export.
- Add backend job queue/events/API then wire Mission Floor and PICO to actual events.

## Exact next development action

1. Execute a baseline using a **new** public RoBERTa text-classifier model on user-supplied labeled data on the Mac. Validate results and report SHA.
2. Generalize ONNX candidate generation from the imported model, keeping the baseline immutable; execute on identical inputs.
3. Add quality and CPU/resource gates, explicit rejection paths, then versioned export and independent reload.

This is a verified source-code checkpoint, **not** evidence that model execution or optimizations were successfully run on the Mac.


---

## M1.1: Immutable Hugging Face revision selection / failed external baseline attempt

- Checkpoint time (Asia/Riyadh): 2026-11-01 21:59:48 +0300
- Branch: `develop/auto-optimizer-v1`; previous verified remote HEAD: `5cbbaad7ee3eeae587dc3d3ce4084807ee34d7a2`
- New HEAD: run `git rev-parse HEAD` after this commit; a commit cannot contain its own SHA.
- Local and remote synchronization: verify `git status -sb`, `git rev-parse HEAD`, and `git ls-remote origin refs/heads/develop/auto-optimizer-v1` after pushing.
- Changed files: `app/nimbo_model_intake_v1.py`, `tools/run_nimbo_baseline_v1.py`, `tests/test_nimbo_model_intake_v1.py`, this checkpoint.
- Verified original run on Mac (user log, 2026-11-01): `cardiffnlp/twitter-roberta-base-sentiment-latest` on default `main` stopped at metadata intake with `No supported safetensors classifier weights`; **no baseline was measured**.
- Observed public repo state: default `main` contained `pytorch_model.bin` but not `model.safetensors`; a historical SHA `b995b77f45ae7ed76649b47d04952ec8be9f3591` has a `model.safetensors` file.
- Implemented: optional explicit immutable `--revision` (40 hex chars); SHA returned by Hub must equal request, tokenizer/config/weights all use resolved SHA; existing safetensors-only, size, trust_remote_code=False and architecture restrictions preserved. Better error for a missing safetensors variant.
- Tests locally conducted in patch authoring container: 32 tests in `tests/test_nimbo_model_intake_v1.py` passed; this is not a Mac hardware inference result.
- Installer executes Mac's full `python -m pytest -q` before committing; observe actual terminal output. No frontend files changed.
- Live inference at selected historical model revision: **NOT YET VERIFIED**. No optimization candidate, export, staging deployment or CPU saving claim.
- Exact next step: execute `python -m tools.run_nimbo_baseline_v1 --model-id cardiffnlp/twitter-roberta-base-sentiment-latest --revision b995b77f45ae7ed76649b47d04952ec8be9f3591 --dataset ~/Downloads/NIMBO_REAL_IMDB_PILOT.csv` on Mac. Inspect metrics and failure logs.
- After measured baseline: build generic ONNX candidate generator and compare on identical inputs/labels.
- GitHub `main` remains protected; **never merge or modify main automatically**.


---

## Milestone 2 — External model ONNX candidate (experimental)

- Updated Asia/Riyadh: 2026-11-01 22:16:39 +0300
- Branch: `develop/auto-optimizer-v1`
- Verified previous GitHub HEAD: `d6d2fbf38b061876406954c719701fa31414f904`
- New checkpoint commit: obtain using `git rev-parse HEAD`; a commit cannot record its own SHA.
- GitHub sync: installer verifies `git ls-remote` before editing; after `--push`, confirm remote SHA. CI branch status is NOT independently verified by this checkpoint.
- `main` hackathon commit unchanged by this milestone.

### Measured user evidence (M1, not M2)

- User verified Mac baseline: `cardiffnlp/twitter-roberta-base-sentiment-latest` at immutable revision `b995b77f45ae7ed76649b47d04952ec8be9f3591`, 12 IMDb demo rows.
- Original model accuracy: **10/12 = 83.33%**; warm forward mean **39.472 ms**, p95 **54.0421 ms**, process CPU forward aggregate **571.227 ms**.
- Report: `results/auto_optimizer_baselines/baseline_7f56528cafaa31bc.json` (Mac-local; ignored by Git).
- Domain mismatch and sample size 12 limit quality conclusions. This does **not** measure optimization or model-size change.

### M2 implementation

- `app/nimbo_onnx_candidate_v1.py`: accepts new public model IDs with M1 security checks and immutable SHA; loads genuine safetensors-only CPU model; exports ONNX FP32 graph with dynamic inputs; uses ONNX Runtime CPUExecutionProvider with optimization enabled.
- `tools/run_nimbo_onnx_candidate_v1.py`: paired PyTorch/ONNX benchmark on identical tokenized inputs; alternating order and 3 rounds by default; tracks predictions, accuracy, agreement, max logits difference, forward mean/p95 and process CPU time separately.
- `tests/test_nimbo_onnx_candidate_v1.py`: tests validation, identity checks, gates and exporter invocation.
- Original M1 source is not modified. No FastAPI routes, frontend, 3D animation, generic INT8 candidate or downloadable runnable model package introduced here.
- A successful execution stores test-only `model.onnx` and `report.json` in ignored `results/auto_optimizer_candidates_v1/<run-id>/`. This is **NOT** an exportable deployable model package; tokenizer/config remain external.
- Local gate requires mean speedup, CPU non-regression, p95 non-regression, equal accuracy, 100% prediction agreement and small numerical error. Samples <50 always exploratory and no candidate is authorized for deployment/export, even if local gates pass.

### Verification scope

- Patch-author unit tests: **29 passed** in sandbox; Mac installer must run full `python -m pytest -q` and report actual result.
- Mac ONNX export + real paired inference: **NOT YET RUN** at checkpoint creation.
- New benchmark is a different measurement from the single-path M1 report and must be treated as such. No quantified savings claim until paired run succeeds.
- Local model execution is not public multi-tenant sandboxing. No electricity, bills, FLOPs or whole-request energy measured.

### Next action

1. Run the M2 CLI with same pinned model SHA and M1 baseline report; inspect ONNX export and full paired metrics.
2. If failure, investigate actual logs and fix on development branch; never fake results.
3. After successful exploratory comparison, introduce a disjoint heldout set, reloadable package with tokenizer/config, and optional genuinely size-reducing INT8 candidate (when hardware supports it).
4. Connect frontend/Backend jobs/3D Lab to real states only after end-to-end candidate pipeline works.


---

## Milestone 3 — M2 ONNX graph reuse + independent-of-M2 heldout evaluation

- Updated Asia/Riyadh: 2026-11-01 22:26:35 +0300
- Branch: `develop/auto-optimizer-v1`
- Pre-install verified GitHub HEAD: `0bea2803bce2dc595d4fbf2c7a650dfb972169b7`. After commit/push, verify the NEW SHA with `git rev-parse HEAD` and `git ls-remote origin refs/heads/develop/auto-optimizer-v1`. A commit cannot contain its own final SHA.
- GitHub CI for this branch: NOT VERIFIED unless actually shown by GitHub.
- Source code on `main`: not modified; hackathon reference historically at `710f4f1`.
- User-reported real local M2 measurement on 12 IMDb pilot rows x 3 rounds: RoBERTa PyTorch vs same weights ONNX, mean warm forward 43.5792 to 28.9344 ms (33.605% reduction), process CPU forward 2101.952 to 1998.066 ms (4.9424% reduction), p95 62.729 to 44.7551 ms; 83.3333% accuracy both, 100% agreement; `EXPLORATORY_PASS_NOT_APPROVED`, not deployed/exportable. Source local report: `results/auto_optimizer_candidates_v1/candidate_3411b428e11e4796/report.json` (not Git tracked). These are user-terminal evidence, not independently verified on patch-author hardware. No smaller-model claim.

### What this M3 patch implements

- Added `app/nimbo_heldout_validation_v1.py`: verify original M2 report + model SHA/revision + original ONNX graph SHA256 before loading graph, then reload ONNX Runtime session in a separate CLI invocation; run PyTorch/ONNX on same 200-800 labeled evaluation rows (3 paired alternating rounds by default), compare accuracy, numerical agreement, warm forward latency/P95 and process CPU, reject CPU/quality regressions and nondeterminism. Output local aggregate JSON only.
- Added `tools/prepare_nimbo_tweeteval_holdout_v1.py`: deterministic selection of REAL public TweetEval sentiment test rows, dataset revision pinned by HF SHA and dataset provenance sidecar. Distinct from M2 IMDb records. Source model's card says it was fine-tuned with TweetEval: this is NOT a claim that the data is independently unseen by model developers.
- Added `tools/run_nimbo_heldout_v1.py` CLI for this independent-of-M2-data test. Optional `--preprocess cardiff_tweet` implements model-card @/URL normalization, recorded in receipt.
- Added `tests/test_nimbo_heldout_validation_v1.py`: unit tests for provenance, safety gates, hash tampering and a MOCKED full workflow.
- No frontend/API integration, multi-tenant sandbox, model package exporter, real INT8, deployment, or agent behavior added.

### Verified tests / restrictions

- Patch-author local tests: 21 passed with mocked model/session (NOT a real RoBERTa model run). Installer runs these and full repository Python regression tests on Mac before committing.
- Real TweetEval download/real 200-row model benchmark: NOT RUN by installer. User must run both documented commands on Mac and inspect measured result.
- Both an M2-local pass and a successful M3 holdout gate are review evidence only; M3 always returns `eligible_for_deployment=false` and `eligible_for_downloadable_package=false`.
- Still no independent package reload with tokenizer/config; no CPU-energy/FLOPs/cloud billing measured.

### Next action

1. Generate real public TweetEval sentiment test CSV using M3 CLI; validate source and checksum.
2. Run M3 CLI against the exact M2 candidate report. Check all gates, reproduce failure details if rejected. Avoid retuning on heldout data.
3. After passing, build self-contained runnable model package + tokenizer/config/manifest + independent reload test (M4); reject package on any failure.
4. Only then connect Auto-Optimizer API/jobs and 3D Lab states; keep `main` untouched until explicit approval.


---

## Milestone 4 — Offline runnable ONNX review package (implementation)

- Updated Asia/Riyadh: 2026-11-01 22:47:50 +0300
- Branch: `develop/auto-optimizer-v1`; verified pre-install remote HEAD `dd88e8adc402443d26d481aa3fd3a7ceaa2e4c37`. Post-commit SHA must be read with `git rev-parse HEAD` and verified against `git ls-remote`.
- GitHub CI after commit: NOT VERIFIED by installer; check run separately. `main` remains untouched.
- User-reported Mac M3 result: model `cardiffnlp/twitter-roberta-base-sentiment-latest` at `b995b77f45ae7ed76649b47d04952ec8be9f3591`; 200 TweetEval TEST examples, 3 paired rounds, accuracy 71% on both paths, mean warm PyTorch 21.0102 ms vs ONNX 8.4126 ms (59.9594% lower), process CPU forward total 15267.9 vs 9392.271 ms (38.4835% lower); 100% prediction agreement. `HELDOUT_LOCAL_PASS_REQUIRES_PACKAGE_RELOAD`; NOT deployment approved.
- Local M3 report: `results/auto_optimizer_heldout_v1/heldout_c5a72a858a014e1f.json` (ignored, not Git tracked). TweetEval data separate from M2 IMDb demo but not proven unseen by original model developer; 200 samples are not an independent third-party audit.

### Code added

- `app/nimbo_package_v1.py`: binds M2/M3 graph SHA, pinned model identity, M3 decision, data fingerprint, preprocessing and 200+ labeled examples; downloads pinned tokenizer/config; creates review-only model ZIP with same ONNX graph, offline inference runner, SHA256 manifest and evidence summary.
- `tools/build_nimbo_package_v1.py`: review package build CLI; outputs only after offline ZIP extraction, new Python subprocess reload and execution of actual 200 labeled examples reproduce M3 candidate accuracy and packaging-time prediction fingerprint; also smoke-tests standalone runner.
- `tools/verify_nimbo_package_v1.py`: verify extracted package hashes, local tokenizer+ONNX inference, and paired prediction fingerprint, without internet.
- `tests/test_nimbo_package_v1.py`: tests for invalid review gate, tampered graph/dataset/manifest, symlinks, rollback and mocked ZIP/reload orchestration.

### Test boundaries

- Patch author: 95 tests passed (M1-M4 combined) with mocked inference in M4; NOT evidence a real RoBERTa package was built or reloaded.
- Installer executes new M4 tests and the full existing Python test suite on Mac before committing. Web frontend unchanged.
- Real packaging/independent unzip reload against M3 data: NOT RUN until the user executes the CLI on Mac. This is a local review-only artifact, not public upload, production deployment, or evidence that ONNX shrank model weights. No electrical power, FLOPs or billing measured.
- M3 report is local self-reported evidence, not cryptographically signed. SHA-256 guards against accidental graph/bundle modification, not malicious replacement of every local record.

### Exact next action

1. Run M4 builder on Mac using M3 report and the same saved TweetEval CSV. Require `STATUS: RUNNABLE_PACKAGE_VERIFIED — REVIEW ONLY` and verify the archive path/SHA.
2. Independently unzip/download and execute `run_model.py --text ...` on a fresh compatible Python environment; confirm offline local operation.
3. Implement candidate selection including actual size-reducing INT8 options, safe worker/background jobs, API import/receipt/export, and only then connect 3D Lab to real backend stages. Do NOT modify `main` without explicit user permission.


---

## Milestone 5 — Dynamic INT8 size-oriented development trial

- Updated (Asia/Riyadh): 2026-11-01 23:02:11 +0300
- Branch: `develop/auto-optimizer-v1`; pre-install verified local+remote HEAD `aef426218d5efc1f3e2c646d03e94de112b356e9`.
- Commit containing this checkpoint: obtain via `git rev-parse HEAD` and compare to `git ls-remote origin refs/heads/develop/auto-optimizer-v1` after installation.
- Historical `main` preserved, previously verified `710f4f1`; this installer never modifies main.
- Branch CI after push: NOT VERIFIED BY THIS INSTALLER. Verify GitHub Actions separately.
- Local Mac artifacts/results can be newer than GitHub, are ignored, and are not uploaded automatically.

### Evidence before M5 (user-shared Mac CLI logs)

- M3 on 200 real TweetEval TEST rows, 3 paired rounds (600 forwards per path), RoBERTa PyTorch FP32 vs ONNX FP32: 71% accuracy on both; mean warm forward 21.0102 vs 8.4126 ms (59.9594% lower); aggregate CPU 15267.9 vs 9392.271 ms (38.4835% lower); 100% prediction agreement. Local review-only, not production approval; dataset not proven unseen during source model training.
- M4: `RUNNABLE_PACKAGE_VERIFIED_REVIEW_ONLY` reported by the user after an offline ZIP extraction and independent-process model reload. Local ZIP `results/auto_optimizer_packages_v1/nimbo_onnx_review_380a8354a0b0.zip`; SHA256 `89df729766ffc9a676724bb95d9a4f4c103af78904c915e8530bc733fec51f9a`; source graph SHA `5163ef3d57227ac163c7f96512a6745d0fd338aed9ff3e680c8c22b5296aa970`; reload accuracy 71%, 200 examples. **Model size reduction not verified**. Reported by Mac user, ZIP not committed to Git.

### M5 code added

- `app/nimbo_int8_candidate_v1.py`: verifies exact prior M4 ZIP SHA, manifest, contents and M3 evidence; converts the SAME packaged ONNX FP32 graph using real ONNX Runtime dynamic INT8 MatMul quantization; validates INT8 operators and independently compares FP32/INT8 inference and physical graph bytes with quality/CPU/latency gates.
- `tools/prepare_nimbo_int8_dev_v1.py`: prepare real 200–800 TweetEval sentiment **validation** split records, source commit pinned, index provenance, SHA. This is NOT the prior M3 TEST split; not proven independently unseen by the original RoBERTa developer.
- `tools/run_nimbo_int8_candidate_v1.py`: development-only benchmark/report CLI, size or balanced policy. No staging or automatic downloadable INT8 package.
- `tests/test_nimbo_int8_candidate_v1.py`: unit/integration-mocked tests for INT8 decision gates, anti-tamper ZIP checks, rejected metrics, dev provenance, fake quantizer orchestration. Mocked tests are NOT real RoBERTa inference.

### Verification and known limits

- Patch-author environment: 120 targeted tests (M1–M5) passed. No ONNX Runtime installed there, so real INT8 model conversion/inference was NOT RUN in the patch-author environment.
- Installer executes M5 tests + full existing Python regression suite and confirms both CLI `--help` imports on Mac before committing. These are NOT proof that quantization on RoBERTa or Apple Silicon will succeed.
- M5 trial measures FP32 ONNX graph bytes versus INT8 ONNX graph bytes, not complete downloadable package, model RAM usage, FLOPs, energy, or cloud billing. Different quantization kernels may be slower or unsupported on some CPUs; failure/rejection is a valid result.
- M5 development set is used for selecting candidates; to promote INT8, evaluate the exact candidate on a separate new or reserved holdout, then bundle/config/tokenizer and independently offline-reload it with identical output verification.
- No UI/mission-floor agents, cloud deployment, security isolation, or production readiness added.

### Exact next action

1. Run M5 dataset-prep CLI (validation split, 200 labeled rows) on Mac.
2. Run M5 INT8 candidate using exact M4 ZIP and printed SHA plus original M3 evidence, check full output and decision; preserve all M1-M4 receipts.
3. If INT8 fails conversion or gates, investigate concrete hardware/errors and reject or explore compatible quantization settings on *development* data; do not retune heldout M3 test.
4. If passes development gates, validate with reserved/new heldout, build independently reloadable INT8 bundle and test again; then implement secure API/job workflow and connect real 3D Lab events. Do not touch `main` without explicit permission.


---

## Milestone 5.1 — Quality-Aware Selective Dynamic INT8 (development only)

- Updated at (Asia/Riyadh): 2026-11-01 23:49:31 +0300
- Branch: `develop/auto-optimizer-v1`; verified pre-install local/remote HEAD: `e5468d0abcaaed5689d3cfc466eb5be0ac734d14`.
- Post-install commit SHA: verify with `git rev-parse HEAD`; a commit cannot contain its own SHA.
- CI status after push: NOT VERIFIED by installer (Python tests are local).
- Local artifacts remain ignored, not synced to GitHub. `main` historically at `710f4f1` remains untouched.

### User-provided Mac evidence for M5

- Exact M5 local report: `results/auto_optimizer_int8_v1/int8_69361d79f93f49e9/report.json`.
- RoBERTa dynamic INT8 graph: 498,831,348 -> 242,447,471 bytes (**51.3969% smaller ONNX graph only**).
- TweetEval validation 200 examples x3; accuracy FP32 .71 -> INT8 .70 (1 pp lower); prediction agreement .925, below unchanged .98 minimum; mean forward 8.8081 -> 6.2656 ms; process CPU forward total 12323.914 -> 9640.785 ms.
- M5 decision `INT8_REJECTED_DEV_GATES`; that rejection is correct. Graph shrink/speed improvement is NOT release permission.

### This patch adds

- `app/nimbo_quality_quantization_v1.py`: small bounded selective dynamic INT8 search using named MatMul node subsets from the exact original M4 FP32 graph. Profiles vary first/last portion of ONNX topological MatMul order and optional per-channel quantization. This is a heuristic; do not call the portions verified semantic transformer layers.
- Rechecks original rejected M5 report, graph SHA, dev dataset SHA, M3 evidence and M4 archive SHA. Reuses M5's unchanged quality, size and runtime gates; records every attempted profile and chooses only an actually passing candidate, otherwise explicit rejection. Maintains FP32 fallback.
- `tools/run_nimbo_quality_int8_v1.py`: development CLI. `tests/test_nimbo_quality_quantization_v1.py`: policy, provenance and MOCKED end-to-end tests.
- All results under ignored `results/auto_optimizer_quality_int8_v1`, no model ZIP exported, no UI, API, or deployment change.

### Verification scope

- Patch-author Python regression: **133 passed** across M1–M5.1 targeted suites with mocked model execution (NOT real ONNXRuntime inference). Confirm installer Mac output for full suite separately.
- New selective quantization on actual RoBERTa hardware: NOT RUN until user invokes CLI on Mac. It may reject all candidates or fail for specific operator versions.
- Actual model file size and speed/accuracy must be measured per candidate, not inferred from profile names. Still no RAM, FLOPs, energy, billing or complete-package size claim.
- Evaluation uses **already used M5 validation split** to choose among candidates. Needs a NEW untouched holdout before an INT8 package can be approved for review. Do not retune the previous TweetEval TEST set.

### Exact next action

1. Run M5.1 CLI with the same pinned M4 ZIP, M3 report/dataset, rejected M5 report, and validation CSV. Inspect `STATUS`, `DECISION` and all profile receipts.
2. If a profile passes, reserve and independently evaluate a new labeled holdout and independently offline-reload any resulting package. Until then `eligible_for_downloadable_package=false`, `eligible_for_deployment=false`.
3. If none pass, keep the verified FP32 ONNX M4 review package. Do not relax fixed gates simply to report a pass.
4. Preserve main; user must explicitly authorize any merge or production deployment.


---

## Milestone 6 — Automatic evidence-bound FP32 fallback decision (review only)

- Date/time (Asia/Riyadh): 2026-11-02 00:01:07 +0300
- Development branch: `develop/auto-optimizer-v1`; pre-install local and GitHub HEAD: `8d7055bcbd75446d270cdd454e91abe02979fcd3`.
- Commit SHA after installation: `git rev-parse HEAD` (cannot embed a commit's own SHA).
- GitHub CI for new SHA: NOT VERIFIED by this installer. Local full Python suite tested by installer.
- Local Mac evidence reports and model weights remain in ignored `results/` and are not uploaded to GitHub.
- Historical `main` SHA `710f4f1` remains unchanged; installer refuses `main`.

### Decision and evidence

- User-reported M5.1 actual Mac result: `results/auto_optimizer_quality_int8_v1/search_13dbb02997d94c39/report.json`.
- Search reported `ALL_SELECTIVE_INT8_CANDIDATES_REJECTED`, `selected=null`.
- All 4 profiles failed unchanged quality gate (prediction agreement minimum .98): `front_50` .945; `back_50` .940; `front_75` .900; `front_50_per_channel` .965.
- Graph sizes ranged 307,896,902–371,779,772 bytes; these are INT8 ONNX graph bytes, **not** total bundle size or accepted artifacts.
- Existing verified ONNX FP32 M4 archive remains the only currently selected *review-only* artifact; quantization is not approved or exported.

### Code added

- `app/nimbo_auto_decision_v1.py`: read-only evidence-driven selector for the *all INT8 rejected* case; validates exact M2–M5.1 evidence chain, M4 SHA and model identity, independently recalculates M5.1 gates, rejects forged/passing/partial reports, extracts the previously verified M4 ZIP, and reruns bundle prediction verification in a fresh Python subprocess with Hugging Face offline mode enabled. Generates a `SELECT_ONNX_FP32_REJECT_INT8` receipt only if offline reload succeeds.
- `tools/choose_nimbo_candidate_v1.py`: CLI for the decision and receipt.
- `tests/test_nimbo_auto_decision_v1.py`: rejection, provenance and mocked orchestration tests.
- Does not rerun INT8 tuning, change source optimization thresholds, create a new model, approve production, modify FastAPI/React, or add agents. The existing M4 FP32 archive is reused intact.

### Verified tests and limits

- Patch author test workspace assembled from M1–M5.1 payloads plus M6: **145 passed** including 12 M6 tests. Earlier model tests use mocked inference; these tests do **not** prove a real RoBERTa M6 run.
- Installer will run M6 unit tests and the full repository Python suite on Mac before committing; inspect terminal results.
- Real M6 subprocess offline reload against M4 ZIP and 200 TweetEval test samples: **NOT YET RUN** on Mac at checkpoint creation.
- A successful M6 receipt means existing ONNX FP32 review package selected, not INT8 selected, not model-size reduction, not deployment.
- TweetEval is not independently proven unseen by original model developer. No FLOPs, energy or cloud cost savings measurement.

### Next action

1. Run `python -m tools.choose_nimbo_candidate_v1` with original M3/M4/M5/M5.1 reports, datasets and exact M4 archive SHA; inspect `STATUS`, receipt and independent offline reload.
2. After measured pass, implement safe backend job orchestration and API: import supported model, track real execution events, persist decision receipts, expose review-only artifact download; then wire existing Mission Floor to those events.
3. Evaluate any new quantization approaches on separate development data only if justified; require a new untouched evaluation and offline reload before package acceptance. Preserve FP32 fallback.
4. Do not merge, overwrite or force-push `main` without explicit user approval.

---

## Milestone 7a — Local auto-optimizer jobs/API (development only)

- Installer date (Asia/Riyadh): 2026-11-02T01:13:48+03:00 (host timezone; verify on Mac).
- Working branch: `develop/auto-optimizer-v1`; verified previous local/remote HEAD: `56129452ac9024384aed45f674bc3b8787a7288f`. New SHA must be read after commit.
- Local Mac and GitHub push status: inspect installer output. No CI run is implied. `main` is untouched.
- User-reported M6 evidence: `VERIFIED_FALLBACK_SELECTED`, `SELECT_ONNX_FP32_REJECT_INT8`, offline 200-sample reload 71% accuracy, four INT8 variants rejected; receipt `selection_d7d6e8b3f9624d04.json` exists on user's Mac, not in Git.
- Added `app/nimbo_auto_pipeline_jobs_v1.py` and `tools/run_nimbo_auto_pipeline_job_v1.py`: local single-worker subprocess job pipeline invoking existing M1 baseline, M2 ONNX paired benchmark, M3 heldout, and M4 independently verified review-only package. Real per-stage event records, persistent status/receipt, fail/reject paths, dataset SHA verification and crash-interrupted state.
- Added `app/nimbo_auto_jobs_api_v1.py` (router mounted via `app/mission_api.py`): `POST /nimbo/auto/jobs`, `GET /nimbo/auto/jobs/{id}`, `/events`, `/package`. Loopback clients only. Verified package hashes required before download; never deploy.
- Inputs: a local supported HF classifier ID at immutable SHA, 2–100 labeled pilot rows placed in `results/auto_optimizer_job_inputs_v1`, and 200–800 labeled evaluation rows in `results/auto_optimizer_heldout_v1`; speed/FP32 ONNX only, `identity`/`cardiff_tweet` preprocessing. No size/balanced orchestration or INT8 automatic choice yet.
- Safety limitations: developer-only localhost service; NOT sandboxed, authentication, multi-user, rate limiting, robust multiworker or cloud job management. No browser UI/Mission Floor wiring in this milestone. No proof source model never saw TweetEval. No cost/power/FLOPs or ONNX graph size reduction claim.
- Tests: installer runs M7 targeted pytest, full Python pytest suite and CLI/API import checks before commit; actual results must be read from installer stdout. Patch author isolated M7 tests used mocked dependencies and did NOT run real RoBERTa through the API.
- Exact next action: run M7 Mac installer; launch development FastAPI on 127.0.0.1:8005; copy existing pilot CSV into designated input folder; submit real job and inspect event/decision/verified package download. Only then connect Mission Floor to true events.
- Preserve historical `main` and all M1–M6 receipts; no merge to main authorized.
