"""Prepare real, pinned TweetEval sentiment *validation* rows for M5 INT8 trials.

NOT the TweetEval test split used for M3. The source model may have been trained
or tuned on this benchmark; do not call this independently unseen training data.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from app.nimbo_model_intake_v1 import REVISION_RE

DATASET_ID='cardiffnlp/tweet_eval'
SEED=20261101


def prepare_dev(dataset, *, revision: str, output: Path, samples: int=200,
                seed: int=SEED) -> tuple[Path,dict]:
    if not REVISION_RE.fullmatch(revision):
        raise ValueError('Dataset must be pinned to an immutable SHA')
    if not 200<=samples<=800 or len(dataset)<samples:
        raise ValueError('Need 200-800 available validation examples')
    output=output.expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    sidecar=output.with_suffix('.source.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError('Refusing to overwrite prepared data or source metadata')
    indexes=sorted(random.Random(seed).sample(range(len(dataset)),samples))
    selected=[]
    counts={'0':0,'1':0,'2':0}
    for ix in indexes:
        row=dataset[ix]
        text,label=row['text'],row['label']
        if not isinstance(text,str) or not 1<=len(text.strip())<=2000 or type(label) is not int or label not in (0,1,2):
            raise ValueError('Invalid source example')
        counts[str(label)]+=1
        selected.append((text,label))
    try:
        with output.open('x',newline='',encoding='utf8') as f:
            writer=csv.writer(f)
            writer.writerow(['text','label'])
            writer.writerows(selected)
        sha=hashlib.sha256(output.read_bytes()).hexdigest()
        provenance={'source':DATASET_ID,'config':'sentiment','split':'validation',
                    'revision':revision,'seed':seed,'rows':samples,
                    'source_indices':indexes,'label_counts':counts,'csv_sha256':sha,
                    'note':'True source validation split; not the M3 test split; not independently unseen to source model developer.'}
        sidecar.write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf8')
    except Exception:
        output.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return output,provenance


def main(argv=None):
    p=argparse.ArgumentParser(description='Prepare genuine TweetEval validation rows for M5; NOT M3 heldout')
    p.add_argument('--samples',type=int,default=200)
    p.add_argument('--output',default='results/auto_optimizer_int8_v1/tweeteval_sentiment_validation_200.csv')
    args=p.parse_args(argv)
    from huggingface_hub import HfApi
    from datasets import load_dataset
    revision=HfApi().dataset_info(DATASET_ID,revision='main').sha
    if not isinstance(revision,str) or not REVISION_RE.fullmatch(revision):
        raise ValueError('Cannot pin TweetEval dataset SHA')
    data=load_dataset(DATASET_ID,'sentiment',split='validation',revision=revision)
    output,info=prepare_dev(data,revision=revision,output=Path(args.output),samples=args.samples)
    print('STATUS: REAL_TWEETEVAL_VALIDATION_PREPARED — DEV ONLY')
    print('DATASET:',output)
    print('DATASET_SHA256:',info['csv_sha256'])
    print('DATASET_REVISION:',revision)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
