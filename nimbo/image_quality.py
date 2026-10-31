"""Optional, measurable CLIP-assisted image quality checker.

Scores are relative to the supplied contrastive candidates, NOT calibrated
probabilities or proof that a required object exists. Keep a human-review path.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Sequence

@dataclass(frozen=True)
class QualityReport:
    passed: bool
    alignment_score: float
    checks: list[dict]
    check_latency_ms: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)

class CLIPImageChecker:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def compare(self, image, descriptions: Sequence[str]) -> list[float]:
        """Return softmax over explicitly provided descriptions, for relative comparison."""
        if len(descriptions) < 2:
            raise ValueError("At least two contrastive descriptions required")
        inp = self.processor(text=list(descriptions), images=image, return_tensors="pt", padding=True, truncation=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with self.torch.inference_mode():
            out = self.model(**inp)
            probs = out.logits_per_image.softmax(dim=-1)[0]
        return [float(v) for v in probs.cpu().tolist()]

    def evaluate(self, image, prompt: str, critical_objects: Sequence[str],
                 *, threshold: float = 0.65) -> QualityReport:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        start = perf_counter()
        # This is a contrastive ranking signal; thresholds require labeled validation.
        scores = self.compare(image, [prompt, "an unrelated random abstract image"])
        alignment = scores[0]
        checks = []
        for obj in critical_objects:
            if not obj.strip():
                continue
            yes = f"a photograph showing a clearly visible {obj}"
            no = f"a photograph with no {obj} visible"
            s = self.compare(image, [yes, no])[0]
            checks.append({"object": obj, "relative_presence_score": round(s, 4), "passed": s >= threshold})
        passed = alignment >= threshold and all(x["passed"] for x in checks)
        reason = "passed_relative_checks" if passed else "low_alignment_or_missing_critical_object"
        return QualityReport(passed, round(alignment, 4), checks,
                             round((perf_counter() - start) * 1000, 2), reason)
