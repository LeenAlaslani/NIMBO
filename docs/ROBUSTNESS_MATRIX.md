# Robustness & adaptation — test matrix

| Scenario | Expected fail-closed behavior | Current proof status |
|---|---|---|
| 800 labeled held-out samples | Accuracy and gates calculated, no false deployment | Successful local rerun 2026-10-31; see evidence |
| No labels, single request | Keep accuracy unverified; never infer from confidence | Source behavior; live demo needed |
| Candidate fails accuracy or p95 gate | REJECTED, not eligible | Unit gate test required as part of full suite |
| Service disconnected | Honest error/offline state, not fabricated progress | Browser integration test needed |
| Image prompt count / relationships | Uncertain remains unverified; no false full-scene certification | Observed v27 umbrella example, more tests needed |
| Image clear actionable FAIL | Escalate to Standard only if evidence/policy warrants, measure extra GPU | Unverified end-to-end |
| Local adoption switch-back | Rollback to original route, preserve audit | Source implementation; integration test needed |

Evidence rule: `feature in code` is NOT `end-to-end passed`. Capture date, run ID, code SHA, expected, actual, screenshot/log in each manual check. Any failure remains a failure in the judge pack.
