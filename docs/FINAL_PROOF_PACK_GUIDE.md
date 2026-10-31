# NIMBO final judging proof pack — Step 18

This step **does not run models**. It reviews the raw Mac evidence already produced and publishes a **sanitized, traceable derived summary** into the review branch.

## Prerequisites

The isolated `~/Desktop/NIMBO_GITHUB_READY_20261031` worktree contains completed Step 13 development+heldout, Step 15 concurrency, Step 16 temporary HTTP, and Step 17 isolated adoption proof logs, including their ONNX artifact lineage.

## Command

```bash
cd ~/Desktop/NIMBO_GITHUB_READY_20261031
~/Desktop/NIMBO/.venv/bin/python -m pytest -q
~/Desktop/NIMBO/.venv/bin/python -m tools.final_judging_pack_v1
```

The tool fails if the **latest** heldout run is not a CPU-time-gated approved-for-review run, if development wasn't locked before heldout, if counts/quality and measured ratios do not agree, or if saved engine/HTTP/adoption evidence does not match the ONNX artifact SHA/model revision/run ID/threads. The tool requires successful isolated adoption and verified rollback. No simulation supplies missing measurements.

On success it updates `README.md` (marked section) and writes:

- `docs/FINAL_JUDGING_EVIDENCE.md`
- `docs/FINAL_JUDGING_EVIDENCE.json`

The script only emits a whitelist of non-private fields, file-relative paths and SHA-256 checksums. It neither publishes model weights nor copies raw logs. The SHA-256 for the growing `independent_validation_v21.jsonl` will change if new records are appended later; regenerate the pack if that happens.

## Scope limits

CPU core-time is `time.process_time()` aggregated across process threads, not electricity or FLOPs. The positive CPU margin is a single small measurement, not statistically certified. The 800 IMDb holdout is separate from *this* development-selection set but was used in earlier NIMBO experiments. Tests with synthetic unlabeled stress requests are not correctness evidence on unseen labels or production capacity validation. The adoption demonstration runs a temporary local service, not a real customer deployment. No main UI end-to-end or energy verification is implied.

## Git workflow after review

Only when the generated summary and `git diff` have been checked and the local test suite passes, stage the updated source code, tests, README and `docs/` (not `results/` raw run logs). Commit on `judging-submission-20261031`, then push that branch to GitHub. Check the green GitHub Actions jobs before merging the pull request into `main`. Never `git push --force` or overwrite deployment state.
