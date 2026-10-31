# NIMBO — cross-domain performance proof (local, review-only)

**Source run:** `heldout_7830c2ff573a47ae`
**Pinned ONNX SHA-256:** `a4d615ada60cd7e9c4869939bf0039c42e717fdad77db0875279ceedaca39c6a`
**Dataset:** TweetEval sentiment TEST, 400 negative + 400 positive; neutral excluded.

| Metric | PyTorch FP32 (4 threads) | ONNX Runtime FP32 (2 threads) |
|---|---:|---:|
| Labeled accuracy | 82.88% | 82.88% |
| Mean warm inference | 10.895 ms | 4.163 ms |
| Mean process CPU time | 15.604 ms | 7.377 ms |

**Measured change:** 61.79% lower warm inference latency and 52.72% lower process CPU time; prediction agreement 100.00%. All predeclared cross-domain review gates passed. **Not deployed.**

**Dataset provenance:** `cardiffnlp/tweet_eval`, pinned commit `b0f6bf8397168a983b87508f0dfd24094eb4e360`, source Parquet SHA-256 `6021687f63cafdc4001dbb11dc9e207b997a17566ff98e739047a9eb686acc03`, selected-row digest `97260cfd078504e07f2fb5e8af12ff7cadee24cc187136cfb1a7ca3f44270094`. Test selection frozen at 800 balanced positive/negative inputs. The raw tweets and local inference logs are deliberately not committed.

**Interpretation:** This is new-domain evidence for the locked *execution configuration*, not evidence that sentiment-classifier accuracy is constant across domains. The original pretrained model may have seen related material. Cross-domain accuracy here is 82.88%, compared with 91.75% on the earlier IMDb evaluation. The three-class task is intentionally reduced to binary, so this is not accuracy on full TweetEval. No third-party audit.

**Limitations:** CPU time is process-wide and is not electricity, FLOPs, cost, or guaranteed compute savings. The trial is local, not deployment. Run `python -m tools.verify_cross_domain_public_v1` to verify published summary consistency without model downloads, and see [`EXTERNAL_DOMAIN_STEP22.md`](EXTERNAL_DOMAIN_STEP22.md) for experiment methodology.
