# NIMBO — Neural Intelligence for Multimodal Budget Optimization

**Smarter AI. Less Compute. Same Ambition.**

HUMAIN Hackathon — Challenge 2: Deliver AI with Less Compute.

## Goal
Implement and evaluate a quality-aware controller for **text classification** and **text-to-image generation**. We aim to reduce inference resources without violating task-specific quality requirements. No performance improvements are claimed before benchmarks.

## Current status
- ✅ Small-first text routing, confidence escalation, event traces (real controller logic)
- ✅ Image fast / quality mode dispatch (backend callbacks)
- ✅ Unit tests and GitHub Actions
- 🧪 Optional text model loading integration (requires suitable matched checkpoints)
- 🧪 Colab T4 experiment for SD1.5 vs LCM-LoRA
- ⬜ Image quality gate, calibrated text thresholds, UI/robot, production backend

**Warning:** `nimbo.demo` uses *synthetic model outputs* only to test control flow. Do not present these as inference benchmarks. The image Colab script runs actual models when executed on a compatible GPU.

## Run locally (no GPU needed)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m nimbo.demo
```

## Architecture
```
Request -> NIMBO Controller -> Task-specific execution backend -> Trace + Result
                       | text: small-first -> confidence gate -> large fallback
                       | image: preview -> few-step diffusion; quality -> baseline
```

## Benchmarks to report
Text: macro F1, accuracy, p50/p95 latency, peak memory, fallback rate.
Image: repeated latency on same T4, peak CUDA allocated memory, prompt adherence and human quality rating. Record any agent overhead and retries.

## Research references
- [Google Imagen (Diffusion)](https://imagen.research.google/)
- [Stability AI — ADD](https://stability.ai/research/adversarial-diffusion-distillation)
- [Hugging Face — LCM-LoRA](https://huggingface.co/docs/diffusers/main/using-diffusers/inference_with_lcm_lora)
- [UC Berkeley — RouteLLM](https://sky.cs.berkeley.edu/project/routellm/)

## Roadmap
1. Choose two text classifiers trained on same labels and evaluate on held-out data.
2. Calibrate confidence threshold, run same-hardware reference and adaptive benchmarks.
3. Run T4 diffusion notebook/script, compare latency + quality.
4. Connect real inference backends via FastAPI, stream traces to animated NIMBO UI.
5. Add Arabic tokenizer efficiency study separately; do not claim Arabic accuracy without Arabic evaluation.
