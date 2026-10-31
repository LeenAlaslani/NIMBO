# Continuous integration and judging submission

## What GitHub Actions verifies

- **Python tests:** installs the small set of required API/test libraries and runs `python -m pytest -q`. These are unit/regression checks with callbacks/mocks; no pretrained-model weights are downloaded.
- **Frontend build:** installs dependencies from `app/web/package-lock.json` using `npm ci`, then builds via `npm run build`.
- **Published summary sanity:** parses `results/judging/LATEST_HELDOUT_EVIDENCE.json` and checks the main recorded fields are internally compatible. It does **not** recalculate latency or model accuracy.

Expected local baseline on 2026-10-31: **119 tests passed** and successful Vite build. Remote CI status is unknown until GitHub executes this workflow on the pushed branch. Fix any CI failures; do not claim green CI before confirmed.

## What GitHub Actions cannot prove

- GPU image generation, VERA's live decision path, browser-to-backend end-to-end correctness, real model checkpoint loading, download reproducibility, ONNX runtime measurements, energy, FLOPs, deployment readiness or commercial billing.
- The platform's historical evaluation used IMDb reference labels on a disjoint development/held-out partition of the same dataset; that is not third-party certified or proof of generalization to arbitrary customers.

## Preparing a reviewable PR

1. Work only in the isolated `NIMBO_GITHUB_READY_20261031` worktree; do not use `git add .` blindly in the original Desktop tree.
2. Run the independent `PREFLIGHT.py` audit (copy/secret patterns), Python tests and frontend build; inspect licensing and any source files likely to contain user data.
3. Stage only reviewed source, tests, public JSON+Markdown judging summaries, documentation, CI workflow, and required lock/config files. Keep raw runtime logs, `.env`, weights, notebooks with private outputs, `node_modules`, and backups out of the repository.
4. Run `git diff --cached --check` and review `git diff --cached --stat`, `git diff --cached --name-only` before any commit.
5. Push the judging **branch**, not `main`; open a PR to `main`; check Actions outcomes and differences. Merge only after manual review. No force push.

## Traceable core result

Latest local run ID: `heldout_cff50858e9a24db2` (2026-10-31). Same model, PyTorch FP32 vs ONNX Runtime FP32 on 800 heldout IMDb examples × 3 paired rounds. Recorded warm CPU mean forward: 28.5501 ms vs 17.7637 ms, accuracy 91.75% vs 91.75%, candidate eligible for manual review only. Quoted latency excludes model setup and warmup. Previous run yielded a different relative speedup.

For exact commands, source caveats, model revision and partition fingerprint, see `docs/REPRODUCE_RESULTS.md`, `docs/JUDGING_CLAIMS.md` and the published evidence JSON/Markdown.
