"""Optional real Hugging Face classifiers. Supply TWO models with the SAME label space.
Requires: pip install 'transformers<5' torch
Before choosing a small model, ensure its confidence is calibrated on validation data.
"""
from __future__ import annotations

def load_classifier(model_id: str, device: int = -1):
    from transformers import pipeline
    pipe = pipeline('text-classification', model=model_id, tokenizer=model_id, device=device)
    def predict(text: str) -> dict:
        pred = pipe(text, truncation=True)[0]
        return {'label': pred['label'], 'confidence': float(pred['score'])}
    return predict
