# NIMBO Step 22 — pre-registered external-domain evaluation

**Purpose:** Evaluate whether the *locked* PyTorch FP32 (4 threads) versus ONNX Runtime FP32 (2 threads) result generalizes to a different **sentiment domain**, not used by NIMBO to select its execution settings. The source IMDb model is NOT fine-tuned in this test.

## Fixed protocol before running

- Source run: `heldout_7830c2ff573a47ae`; verify its original review gates, pinned model revision, staged ONNX SHA-256, and candidate CPU thread count.
- New dataset: [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval), `sentiment` config, **test** split, pinned official Parquet repository revision `b0f6bf8397168a983b87508f0dfd24094eb4e360`, path `sentiment/test/0000.parquet`. The original script-based pin `55864a3201f347e9edfac045e7f772eb2c0ee801` could not load with current `datasets`; the replacement was chosen **before any test predictions or outcomes were observed**. Dataset file SHA-256 is recorded in the output.
- Fixed seed `20261031`; choose **400 negative and 400 positive examples**, with deterministic balanced sampling; exclude **neutral** tweets because this IMDb classifier predicts only two classes.
- Same input texts/tokenizer/model weights and same fixed order alternation per example. Three CPU rounds, first three warmups excluded from forward timing; first-round gold labels determine accuracy; CPU time counts process threads.
- Predeclared gates: agreement ≥99.5%, accuracy drop ≤1 percentage point, mean forward wall-time improvement ≥5%, **process CPU time improvement ≥2%**, no p95 regression, faster/less CPU in every round, and *diagnostic* absolute accuracy ≥75% on both paths.
- The 75% floor is a deliberately predeclared deployment-readiness diagnostic, not a universal SLA or evidence that an IMDb model is well suited to Twitter.
- **Do not retune, repeat until a good number appears, or add a failed attempt to the main success claim.** A failure is an informative result about domain shift. The output path refuses overwrites.

## Run (in isolated judging worktree)

```bash
cd ~/Desktop/NIMBO_GITHUB_READY_20261031
~/Desktop/NIMBO/.venv/bin/python -m tools.external_domain_v1
```

An initial dataset download may be required. No dataset text, OAuth credential, or model weights are committed by the tool; the aggregate report is under `results/external_domain_v1/` (ignored by Git).

## Interpretation and limitations

This is **new-to-NIMBO cross-domain evaluation**, not an independent external audit. The pretrained model may have been exposed to related text during original training. TweetEval is short social-media sentiment, not customer support and not full 3-class sentiment performance; excluding neutral changes the test population. Lower process CPU time does not imply lower energy, FLOPs, hardware cost, or cloud bill. Shared process peak RSS is not a per-model memory delta. All inference is local CPU (not HTTP or end-to-end application). The 800-row IMDb holdout has been repeatedly inspected historically; this distinct domain is a useful challenge but not a full independent trial. The output never deploys a model.

## Step 22A — compatibility repair (before first evaluation)

The first invocation stopped in `load_dataset` with `RuntimeError: Dataset scripts are no longer supported, but found tweet_eval.py`. **No sentiment samples were evaluated and no candidate performance was observed.**

To work with current `datasets`, the loader now downloads an official TEST Parquet file from `cardiffnlp/tweet_eval` at the immutable commit above and loads it via `load_dataset('parquet', data_files=..., split='test')`. The dataset repository revision changed from the legacy script snapshot to an official Parquet snapshot; exact byte-for-byte equality of these two upstream historical revisions has *not* been established. The fixed seed, binary class mapping, sample sizes, model/artifact, CPU thread counts, gates and 3-round protocol did not change. This is the first valid external-domain evaluation, not a rerun to seek a better metric.

A source file SHA-256 and selected-row digest are written into the local report. No data samples or model weights are committed.
