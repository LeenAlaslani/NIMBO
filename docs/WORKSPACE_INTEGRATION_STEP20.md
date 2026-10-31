# Step 20 — Main NIMBO Workspace/API/Evidence contract and browser walkthrough

## What is actually tested

Run from `~/Desktop/NIMBO_GITHUB_READY_20261031` using the **existing running original NIMBO backend and Vite frontend** on ports 8000 and 5173:

```bash
~/Desktop/NIMBO/.venv/bin/python -m tools.workspace_integration_v1
```

The probe verifies: FastAPI health and policy presence; comparison and heldout catalogs; Vite React root and the actually served Workspace/Mission Floor/Evidence integration routes; a real POST to `POST /nimbo/bench/compare` using two **unlabeled** requests; all NDJSON stages; an actual result, measured times, and `UNVERIFIED` guardrail; persistent aggregate comparison record; and that the unified Evidence Center API includes the record with `quality_certified=false`.

**Side effect:** The probe writes **one aggregate metadata record** to `results/baseline_comparisons_v15.jsonl` in the **running original NIMBO backend**. It does not save prompt texts, modify optimization policies, deploy variants, or activate canaries. It does not change source code of original NIMBO. The test itself is not the 800-example validation and cannot justify any accuracy or compute savings claim.

Output record (local and ignored by Git): `results/workspace_integration_v1/workspace_*.json` inside the **isolated GitHub worktree**. If `STATUS: FAIL`, retain and report the exact message. Do not falsely mark it passed.

## Browser walkthrough (manual, required to cover actual UI behavior)

1. Open `http://127.0.0.1:5173` with the existing original NIMBO services still running. Check the Workspace loads instead of a blank page.
2. Choose **Customer feedback classification**, **Paired baseline comparison**, and **Single request**. Enter a harmless example sentence and leave **Expected sentiment** set to **Not labeled**.
3. Click **Enter Mission Floor**, then follow PICO through **CAPTAIN → TOKI → ROUTA → MAKO → VERA → METRI**. At ROUTA, verify a real backend request starts; wait for the real result and event stages. Robot animations can include presentation delays and do not measure inference time.
4. At VERA, expect **UNVERIFIED / Accuracy not evaluated**, because no reference label was submitted. This is correct behavior, not a bug. Confirm METRI reports measured latency without energy/FLOP claims.
5. Open **Evidence Center**, click **Refresh evidence**, and find a new **Paired CPU comparison** record. It must not be marked as independently quality certified. Compare the response and logged values.
6. Visit Fleet Control/Model Registry and confirm they display configured inventory honestly, not an assertion that every model is already loaded and healthy.

**Results boundary:** A successful Python probe proves live HTTP integration contracts and evidence persistence. A successful manual walkthrough additionally confirms the browser interaction and 3D flow. Neither provides independent UI automation, production-scale reliability, external heldout generalization, energy or billing evidence.
