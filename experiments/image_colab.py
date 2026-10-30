"""Run in Google Colab T4. Requires HF model access, internet, and diffusers stack.
Install in Colab: !pip install -q diffusers transformers accelerate safetensors peft
Run from a GPU-enabled session, not on CPU. This is a starter, not benchmark results.
"""
import time
import torch
from diffusers import StableDiffusionPipeline, LCMScheduler

MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
LORA = 'latent-consistency/lcm-lora-sdv1-5'
PROMPT = 'a minimalist futuristic library interior, natural light, architectural photography'

pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
pipe = pipe.to('cuda')
pipe.enable_attention_slicing()

def generate(steps, guidance, seed=42):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    image = pipe(PROMPT, num_inference_steps=steps, guidance_scale=guidance,
                 width=512, height=512, generator=torch.Generator('cuda').manual_seed(seed)).images[0]
    torch.cuda.synchronize()
    return image, {'latency_s': round(time.perf_counter()-start,3),
                   'peak_allocated_gb': round(torch.cuda.max_memory_allocated()/1e9,3)}

# Warm-up each configuration before actual benchmark runs.
reference, reference_stats = generate(25, 7.5)
reference.save('reference.png')
print('Reference:', reference_stats)

pipe.load_lora_weights(LORA)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
fast, fast_stats = generate(4, 1.0)
fast.save('fast.png')
print('LCM LoRA:', fast_stats)
print('Important: benchmark with multiple prompts, repeated runs, same GPU, quality review.')
