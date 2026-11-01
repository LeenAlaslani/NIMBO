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
