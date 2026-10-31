# GitHub publishing preparation

Existing public repository: https://github.com/LeenAlaslani/NIMBO (`main`). Its last inspected state contains an older README and limited source, so **preserve existing Git history and make a new review branch**. This step-two package **does not publish or change GitHub**.

Before publishing: remove local `.venv`, `node_modules`, `dist`, backups, caches, model weight binaries, user images, all `.env*`, tokens, private datasets, machine logs and sensitive identifiers. Review tracked file list; `.gitignore` does not remove already-tracked files. Confirm licenses/terms for third-party models and code.

Recommended workflow on the Mac (only after review):

```bash
cd ~/Desktop/NIMBO
git status --short
git remote -v
```

If no `.git` exists, **do not blindly run `git init && git push --force`**. Clone the existing repo in a separate folder, create a review branch, copy the reviewed project selectively, run tests, then commit and open a PR. Preserve the old main branch until the PR is reviewed.

A judge should find these files immediately: `README.md`, `docs/JUDGING_CLAIMS.md`, `docs/REPRODUCE_RESULTS.md`, `results/judging/LATEST_HELDOUT_EVIDENCE.md`, source files, regression tests, and a GitHub Actions smoke workflow. Source-only review does not guarantee weights or GPU are available at judge runtime.
