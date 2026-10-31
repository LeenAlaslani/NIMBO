"""Optional real GPU image generation + CLIP relative-score checker.
Requires local SD1.5 / LCM LoRA weights or network access to download.
Do not run on Mac 16GB: intended for CUDA/T4.
"""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
from nimbo.recovery_generation_v9 import run


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prompt',required=True)
    ap.add_argument('--object',dest='critical_object',required=True)
    ap.add_argument('--seed',type=int,default=134)
    ap.add_argument('--budget',type=float,default=45)
    ap.add_argument('--steps-fast',type=int,default=4)
    ap.add_argument('--steps-standard',type=int,default=25)
    ap.add_argument('--output',default='results/recovery_generation_v9')
    args=ap.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This real image generation script requires CUDA GPU (Colab T4).')
    from diffusers import StableDiffusionPipeline,LCMScheduler
    from nimbo.image_quality import CLIPImageChecker
    from PIL import Image
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    # A single SD1.5 pipeline for standard calls, and an LCM LoRA-enabled copy for fast.
    # Loading two models can cause T4 OOM; use one pipeline and switch adapters/schedulers.
    base='runwayml/stable-diffusion-v1-5'
    pipe=StableDiffusionPipeline.from_pretrained(base,torch_dtype=torch.float16,safety_checker=None).to('cuda')
    pipe.enable_attention_slicing()
    original_scheduler=pipe.scheduler
    pipe.load_lora_weights('latent-consistency/lcm-lora-sdv1-5',adapter_name='lcm')
    pipe.set_adapters(['lcm'],adapter_weights=[1.0])
    lcm_scheduler=LCMScheduler.from_config(original_scheduler.config)
    verifier=CLIPImageChecker(device='cpu')
    counter=0
    def generate(stage,prompt,index):
        nonlocal counter
        if stage=='fast':
            pipe.set_adapters(['lcm'],adapter_weights=[1.0]);pipe.scheduler=lcm_scheduler
            steps,guidance=args.steps_fast,1.0
        else:
            pipe.disable_lora();pipe.scheduler=original_scheduler
            steps,guidance=args.steps_standard,7.5
        seed=args.seed+index  # retry changes random seed instead of repeating failed image
        gen=torch.Generator(device='cuda').manual_seed(seed)
        image=pipe(prompt,num_inference_steps=steps,guidance_scale=guidance,
                   generator=gen,height=512,width=512).images[0]
        fname=root/f'{index:02d}_{stage}.png';image.save(fname)
        counter+=1
        return image
    def check(im,prompt):
        assessment=verifier.evaluate(im,prompt,[args.critical_object],threshold=.65)
        return ('PASS' if assessment.passed else 'FAIL',assessment.reason)
    result=run(args.prompt,generate,check,critical_object=args.critical_object,budget_s=args.budget,max_attempts=3)
    meta={'status':result.status,'checker_pass':result.checker_pass,'elapsed_s':result.elapsed_s,
          'attempts':[asdict(a) for a in result.attempts],
          'disclaimer':'CLIP relative scores are NOT calibrated full prompt verification. No human labels involved in routing.'}
    (root/'trace.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
