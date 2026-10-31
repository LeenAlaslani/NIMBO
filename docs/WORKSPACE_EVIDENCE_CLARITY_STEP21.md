# NIMBO Step 21 — Workspace evidence clarity

This is a **presentation-only** fix grounded in the October 31 screenshot review. It does not modify model inference, timing, policies, quality gates, or evidence log writes.

- The Mission Floor stage counter says `WORKER TOUR`, not `MISSION`, when a completed backend result is being inspected; the visible worker progression is a guided review, not proof of multi-agent inference happening sequentially.
- METRI receipt shows `Paired baseline comparison` rather than breaking `paired_baseline_candidate` across lines.
- The mission wall shows `48.0%` with the already explicit `Lower candidate latency` caption; avoids unsupported glyphs in 3D fonts and displays `Not measured` when the reduction is missing.
- Evidence Center surfaces the latest persisted **workspace paired comparison**, selected by its record timestamp rather than collector order. Its Details action opens the actual recorded fields. It explains why `Live API Requests` does not increase for this evidence class.
- A one-example BERT/DistilBERT comparison is **not** the same-model, 800-example ONNX proof. No new accuracy, compute, energy, or billing claims.

Test with `node --test app/web/tests/evidence-presentation.test.mjs`, `cd app/web && npm run build`. Then reload browser and repeat the already supplied positive-customer-support example. Backend code is unchanged.
