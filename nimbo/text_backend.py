"""Real Hugging Face text classifiers with normalized SST-2 class names."""
from __future__ import annotations


def normalize_sst2_label(label: str) -> str:
    mapping = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE", "NEGATIVE": "NEGATIVE", "POSITIVE": "POSITIVE"}
    if label.upper() not in mapping:
        raise ValueError(f"Unexpected SST-2 label: {label}")
    return mapping[label.upper()]


def load_classifier(model_id: str, device: int = -1, *, sst2_labels: bool = False):
    from transformers import pipeline
    classifier = pipeline("text-classification", model=model_id, tokenizer=model_id, device=device)
    def predict(text: str) -> dict:
        output = classifier(text, truncation=True)[0]
        label = normalize_sst2_label(output["label"]) if sst2_labels else output["label"]
        return {"label": label, "confidence": float(output["score"])}
    return predict
