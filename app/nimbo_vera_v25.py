"""NIMBO VERA v25: real open-vocabulary detection, conservative decisions.

OWLv2 is a real detector (no synthetic inference). The detected boxes and
measured processing time are evidence, not independent quality certification.
Unsupported semantics and low-confidence cases REQUIRE human review.
"""
from __future__ import annotations
import base64
import io
import math
import os
import re
import threading
import time
from functools import lru_cache

MAX_IMAGE_B64 = 11_000_000
MODEL_ID = 'google/owlv2-base-patch16-ensemble'
COLORS = {'red','orange','yellow','green','blue','purple','pink','black','white','brown','gray','grey'}
COUNT_WORDS = {'a':1,'an':1,'one':1,'single':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
# Deliberately bounded supported nouns: unknown or multiple object types never auto-accepted.
NOUN_ALIASES = {
    'umbrella':['umbrella','parasol'], 'cat':['cat'], 'dog':['dog'],
    'lifebuoy':['lifebuoy','life ring','life buoy'], 'apple':['apple'],
    'mug':['mug','coffee mug'], 'sunflower':['sunflower'], 'backpack':['backpack'],
    'teapot':['teapot'], 'bicycle':['bicycle','bike'], 'sailboat':['sailboat','sailing boat'],
    'taxi':['taxi','taxi cab'], 'flag':['flag'], 'train':['train'],
    'greenhouse':['greenhouse'], 'car':['car'], 'boat':['boat'],
    'bird':['bird'], 'flower':['flower']
}
SCENE_TERMS = ('outside','inside','rain','rainy','street','road','grass','garden','café','cafe',
               'sea','ocean','lake','field','sidewalk','bridge','sky','room','on a','in a',
               'near','beside','behind','next to','hanging','holding','wearing','photograph','realistic')
_MODEL_LOCK=threading.RLock()


def analyze_prompt(prompt):
    p=' '.join(prompt.lower().replace('’',"'").split())
    mentioned=[]
    for canonical, aliases in NOUN_ALIASES.items():
        if any(re.search(r'\b'+re.escape(alias)+r'(?:s|es)?\b',p) for alias in aliases):
            mentioned.append(canonical)
    # Only unambiguous single object types can be auto-accepted. Other object types
    # may be contextual (e.g. cafe/street) but remain unverified, so human review.
    if not mentioned:
        return {'supported':False,'reason':'No supported critical object found','object':None,
                'required_count':None,'required_color':None,'scene_unverified':True}
    primary=mentioned[0]
    pos=re.search(r'\b(?:(\d+|a|an|one|single|two|three|four|five|six|seven|eight|nine|ten)\s+)?(?:(red|orange|yellow|green|blue|purple|pink|black|white|brown|gray|grey)\s+)?'+re.escape(primary)+r'(?:s|es)?\b',p)
    # Preserve safety for unusual grammar: no automatic acceptance without parsing the requirement.
    if not pos:
        return {'supported':False,'reason':'Cannot unambiguously parse number/color','object':primary,
                'required_count':None,'required_color':None,'scene_unverified':True}
    raw_count=pos.group(1)
    count=int(raw_count) if raw_count and raw_count.isdigit() else COUNT_WORDS.get(raw_count,1)
    color=pos.group(2)
    scene=any(s in p for s in SCENE_TERMS)
    return {'supported':len(mentioned)==1,'reason':'Additional critical objects present' if len(mentioned)>1 else None,
            'object':primary,'required_count':count,'required_color':color,
            'scene_unverified':scene,'other_objects':mentioned[1:],
            'has_explicit_count':raw_count is not None}


def _base64_image(encoded):
    if not isinstance(encoded,str) or len(encoded)>MAX_IMAGE_B64:
        raise ValueError('Image missing or exceeds local verifier upload limit')
    if encoded.startswith('data:image/png;base64,'):
        encoded=encoded.split(',',1)[1]
    raw=base64.b64decode(encoded,validate=True)
    if not raw.startswith(b'\x89PNG\r\n\x1a\n') or len(raw)>8_000_000:
        raise ValueError('Only PNG image bytes from the local worker are supported')
    from PIL import Image
    image=Image.open(io.BytesIO(raw)); image.load()
    if image.width>2048 or image.height>2048 or image.width<64 or image.height<64:
        raise ValueError('Unexpected image dimensions')
    return image.convert('RGB'),raw


def _device():
    import torch
    preference=os.getenv('NIMBO_VERA_DEVICE','cpu').lower()
    if preference=='mps' and torch.backends.mps.is_available(): return 'mps'
    if preference=='cuda' and torch.cuda.is_available(): return 'cuda'
    return 'cpu'  # CPU avoids competing for 16GB Mac's shared GPU memory.


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    device=_device()
    processor=Owlv2Processor.from_pretrained(MODEL_ID, use_fast=False)  # use established PIL-based preprocessing
    model=Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
    return processor,model,device


def _iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0)
    area=lambda z:max(0,z[2]-z[0])*max(0,z[3]-z[1])
    union=area(a)+area(b)-inter
    return inter/union if union>0 else 0.


def non_max_suppression(items,threshold=.4,max_boxes=15):
    """Deduplicate proposals across multiple noun synonyms, ordered by confidence."""
    dedup=[]
    for item in sorted(items,key=lambda x:x['score'],reverse=True)[:100]:
        if all(_iou(item['box'],other['box'])<threshold for other in dedup):
            dedup.append(item)
            if len(dedup)>=max_boxes:break
    return dedup


def color_fraction(image, box, color):
    """Simple pixel hue test *inside detector box*; contextual background can confuse it.

    Not a learned color classifier. Used as a veto, never as proof of the scene.
    """
    import numpy as np
    from PIL import Image
    x0,y0,x1,y1=[int(v) for v in box]
    x0=max(0,x0);y0=max(0,y0);x1=min(image.width,x1);y1=min(image.height,y1)
    if (x1-x0)<10 or (y1-y0)<10: return 0.
    patch=image.crop((x0,y0,x1,y1)).resize((80,80))
    a=np.asarray(patch.convert('HSV'),dtype=np.uint8).astype('float32')
    hue,sat,val=a[:,:,0],a[:,:,1]/255,a[:,:,2]/255
    if color in ('black',): mask=val<.24
    elif color=='white':mask=(val>.78)&(sat<.24)
    elif color in ('gray','grey'):mask=(sat<.16)&(val>.22)&(val<.84)
    elif color=='brown':mask=((hue<32)|(hue>245))&(sat>.26)&(val<.72)
    else:
        ranges={'red':((hue<=12)|(hue>=244)), 'orange':((hue>=10)&(hue<=27)),
                'yellow':((hue>=25)&(hue<=44)),'green':((hue>=45)&(hue<=112)),
                'blue':((hue>=113)&(hue<=177)),'purple':((hue>=178)&(hue<=223)),
                'pink':((hue>=218)|(hue<=9))}
        mask=ranges.get(color,np.zeros_like(hue,dtype=bool))&(sat>.35)&(val>.18)
    # Center crop reduces context/background but may skip thin objects.
    center=mask[8:72,8:72]
    return round(float(center.mean()),4)


def _detect(image, object_name):
    # NIMBO_VERA_V25_1_COMPAT_REPAIR: stable slow processor and explicit error stages.
    stage='torch_import'
    try:
        import torch
        stage='model_load'
        processor,model,device=_load()
        phrases=['a photo of a '+x for x in NOUN_ALIASES[object_name]]
        stage='image_preprocess'
        inputs=processor(text=[phrases],images=image,return_tensors='pt')
        inputs={k:v.to(device) for k,v in inputs.items()}
        if device=='mps':torch.mps.synchronize()
        stage='model_forward'
        start=time.perf_counter()
        with torch.inference_mode():output=model(**inputs)
        if device=='mps':torch.mps.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        stage='box_postprocess'
        # Documented API: image processor owns generic detection postprocessing.
        # Older versions also expose a delegated method on the parent processor.
        image_processor=getattr(processor,'image_processor',None)
        post=getattr(image_processor,'post_process_object_detection',None)
        if not callable(post):
            post=getattr(processor,'post_process_object_detection',None)
        if not callable(post):
            raise RuntimeError('OWLv2 processor lacks object detection postprocess method')
        results=post(output,threshold=.10,
                     target_sizes=torch.tensor([[image.height,image.width]]))
        result=results[0]
        stage='decode_boxes'
        detections=[]
        for score,box in zip(result['scores'],result['boxes']):
            x=[round(float(v),2) for v in box.detach().cpu().tolist()]
            detections.append({'score':round(float(score.detach().cpu()),4),'box':x})
        return non_max_suppression(detections),round(elapsed,2),device
    except Exception as exc:
        try:exc.nimbo_vera_stage=stage
        except Exception:pass
        raise


def verify(image_b64,prompt,detector=None):
    """Return measured automatic triage. Never imply certified success."""
    started=time.perf_counter()
    spec=analyze_prompt(prompt)
    summary={'model':MODEL_ID,'policy':'conservative_development_v25',
             'verdict':'UNCERTAIN','route_advice':'human_review','spec':spec,
             'independently_audited':False,'quality_certified':False,
             'note':'OWLv2 is a fallible zero-shot detector. Detection is not a certified full-prompt quality audit.'}
    if not spec.get('supported'):
        summary['reason']=spec.get('reason') or 'Cannot safely check all critical objects'
        summary['verification_ms']=round((time.perf_counter()-started)*1000,2)
        return summary
    image,_=_base64_image(image_b64)
    try:
        with _MODEL_LOCK: found,inference_ms,device=(detector or _detect)(image,spec['object'])
    except Exception as exc:
        summary.update({'reason':'Automatic model unavailable or failed; manual review required',
                        'error_type':type(exc).__name__,
                        'error_stage':getattr(exc,'nimbo_vera_stage','unknown'),
                        'error_detail':' '.join(str(exc).split())[:260],
                        'verification_ms':round((time.perf_counter()-started)*1000,2)})
        return summary
    plausible=[d for d in found if d['score']>=.16]
    reliable=[d for d in found if d['score']>=.50]
    for d in plausible:
        d['requested_color_fraction']=(color_fraction(image,d['box'],spec['required_color'])
             if spec['required_color'] else None)
    color_ok=(not spec['required_color'] or all(d.get('requested_color_fraction',0)>=.18 for d in reliable[:spec['required_count']]))
    summary.update({'detections':plausible[:12],'model_inference_ms':inference_ms,'device':device,
                    'boxes_plausible':len(plausible),'boxes_reliable':len(reliable),
                    'color_check':'pass' if color_ok and reliable else 'uncertain',
                    'scene_check':'requires_human_or_separate_semantic_verifier' if spec['scene_unverified'] else 'not_required'})
    # Reliable EXPLICIT COUNT mismatch can be escalated, but only when at least one
    # confident detection exists and the requested count is larger. Detector misses
    # are possible: this is a conservative escalation, not proof of failure.
    if spec['required_count']>=2 and len(reliable)>=1 and len(plausible)<spec['required_count']:
        summary.update({'verdict':'FAIL','route_advice':'fallback','reason':'Strong localization found fewer than requested items; escalate, not a definitive absence proof'})
    elif len(reliable)>=spec['required_count'] and color_ok and not spec['scene_unverified'] and spec['required_count']==1:
        summary.update({'verdict':'PASS','route_advice':'accept_fast','reason':'High-confidence primary object and requested color checked; no additional scene requirements parsed'})
    elif spec['required_color'] and reliable and not color_ok:
        summary.update({'verdict':'UNCERTAIN','reason':'Detected object but color evidence is weak; manual review required'})
    else:
        summary.update({'verdict':'UNCERTAIN','reason':'Count, scene, or detection confidence is not sufficiently established'})
    summary['verification_ms']=round((time.perf_counter()-started)*1000,2)
    return summary
