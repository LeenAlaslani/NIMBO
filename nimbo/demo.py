"""A no-GPU smoke-test. Synthetic callbacks DO NOT produce benchmark claims."""
from dataclasses import asdict
import json
from nimbo.core import NimboAgent, Request, Modality

agent = NimboAgent(
    text_small=lambda text: {'label':'general', 'confidence':0.52 if 'ambiguous' in text else 0.92},
    text_large=lambda text: {'label':'support', 'confidence':0.96},
    image_fast=lambda prompt: {'status':'image_backend_not_configured','prompt':prompt},
    image_quality=lambda prompt: {'status':'image_backend_not_configured','prompt':prompt},
)
for req in [Request(Modality.TEXT, 'Where is my order?'),
            Request(Modality.TEXT, 'an ambiguous complaint'),
            Request(Modality.IMAGE, 'a futuristic waterfront city', 'preview')]:
    print(json.dumps(asdict(agent.run(req)), ensure_ascii=False, indent=2))
