"""Download genuine public TweetEval SENTIMENT test rows for M3 local evaluation.

This is a same-task test split but NOT independently proven unseen by the public
model developer. Separate from our M2 IMDb development dataset.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

DATASET_ID = 'cardiffnlp/tweet_eval'
SEED = 20261101


def prepare(dataset, *, revision: str, output: Path, samples: int=200, seed: int=SEED) -> tuple[Path,dict]:
    if not 200 <= samples <= 800:
        raise ValueError('Select 200-800 test rows')
    if len(dataset) < samples:
        raise ValueError('Source test set too small')
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(dataset)),samples))
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() or output.with_suffix('.source.json').exists():
        raise FileExistsError('Dataset and/or provenance file already exists; refuse overwrite')
    # Validate all source rows before creating the destination file.
    counts={'0':0,'1':0,'2':0}
    selected=[]
    for index in indexes:
        row=dataset[index]
        text=row['text']
        label=int(row['label'])
        if not isinstance(text,str) or not 1 <= len(text.strip()) <= 2000 or label not in (0,1,2):
            raise ValueError(f'Invalid official source row at index {index}')
        counts[str(label)]+=1
        selected.append((text,label))
    with output.open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['text','label'])
        writer.writerows(selected)
    meta={'source':DATASET_ID,'config':'sentiment','split':'test','revision':revision,
          'seed':seed,'rows':samples,'source_indices':indexes,'label_counts':counts,
          'csv_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
          'note':'Real public TweetEval rows, disjoint from NIMBO M2 IMDb trial by source. The original model is described as finetuned using TweetEval; do not claim training-independent external validation.'}
    output.with_suffix('.source.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    return output,meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/auto_optimizer_heldout_v1/tweeteval_sentiment_test_200.csv')
    p.add_argument('--samples',type=int,default=200)
    args=p.parse_args()
    from huggingface_hub import HfApi
    from datasets import load_dataset
    revision=HfApi().dataset_info(DATASET_ID, revision='main').sha
    if not isinstance(revision,str) or len(revision)!=40:
        raise ValueError('Cannot pin source dataset revision')
    records=load_dataset(DATASET_ID,'sentiment',split='test',revision=revision)
    file,meta=prepare(records,revision=revision,output=Path(args.output),samples=args.samples)
    print('STATUS: REAL_PUBLIC_TEST_ROWS_PREPARED — NOT MODEL_VALIDATION')
    print('DATASET:',file)
    print('SOURCE:',DATASET_ID,'sentiment test @',revision)
    print('SHA256:',meta['csv_sha256'])


if __name__=='__main__':
    main()
