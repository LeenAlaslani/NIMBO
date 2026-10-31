# Step 17 — Real NIMBO local adoption HTTP probe

Run from the **isolated GitHub worktree**, not the running original NIMBO application:

```bash
~/Desktop/NIMBO/.venv/bin/python -m tools.adoption_proof_v1
```

The CLI checks the latest exported ONNX candidate, exact SHA-256, pinned revision,
heldout CPU objective, and selected development thread count. It starts one
**temporary loopback FastAPI server** hosting NIMBO's actual `app.nimbo_adoption_v21_1`
router, but points only its state/event/request paths at an isolated
`results/adoption_proof_v1/session_*/` directory. Ports 8000/8011/5173 are
unaffected. The server is terminated at the end.

Sequence: baseline response → opt-in Shadow → six paired Shadow classifications
with baseline served → attempt Canary only if NIMBO's own promotion gate permits
→ seven diagnostic requests (first five guarded pairs) → explicit rollback to
PyTorch → confirm baseline serving. All calls travel over localhost HTTP.

`STATUS: PASS` requires completing all stages. `GATE_BLOCKED` is a safe decision,
not a successful end-to-end Canary. Any mismatch, auto-rollback, missing staged
artifact or invalid route fails the proof. The runner attempts rollback even on
error, and never changes the original adoption state or production routes.

**Evidence:** `results/adoption_proof_v1/session_*/report.json` and companion
`events.jsonl`, `requests.jsonl`, `server.log`, `state.json`. No raw request text
is retained by NIMBO adoption logging. Do not publish full local logs blindly.

**Limits:** diagnostic inputs are unlabeled and small, so this does not establish
classification accuracy, real-world reliability, energy savings or cloud cost.
Shadow runs both engines and costs *more*, not less. Canary-only requests cannot
independently verify prediction agreement. This is not the actual Vite UI or
production NIMBO API on port 8000.
