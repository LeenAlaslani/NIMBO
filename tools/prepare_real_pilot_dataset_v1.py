"""Create a genuine labeled company-pilot CSV from the public IMDb TEST dataset.

Reviews are downloaded locally from the source. Not bundled or invented, and
this small product demo is NOT a new independent benchmark or certification.
"""
from __future__ import annotations
import csv,hashlib,json,random
from pathlib import Path

SOURCE='stanfordnlp/imdb'
OUTPUT=Path.home()/'Downloads'/'NIMBO_REAL_IMDB_PILOT.csv'
SEED=20261031

def prepare(dataset=None,output=OUTPUT):
    if dataset is None:
        from datasets import load_dataset
        dataset=load_dataset(SOURCE,split='test')
    rng=random.Random(SEED)
    indexes=list(range(len(dataset)));rng.shuffle(indexes)
    selected={0:[],1:[]}
    for index in indexes:
        row=dataset[index];label=int(row['label']);text=str(row['text']).strip()
        if label in selected and len(selected[label])<6 and 80<=len(text)<=1800:
            selected[label].append((index,text,label))
        if all(len(v)==6 for v in selected.values()):break
    if not all(len(v)==6 for v in selected.values()):
        raise ValueError('Could not find 6 genuine short IMDb TEST reviews per sentiment')
    rows=selected[0]+selected[1]
    rng.shuffle(rows)
    output=Path(output).expanduser();output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['text','label'])
        for _,text,label in rows:w.writerow([text,'POSITIVE' if label==1 else 'NEGATIVE'])
    meta={'dataset':SOURCE,'split':'test','rows':len(rows),'source_row_indices':[int(r[0]) for r in rows],
          'seed':SEED,'note':'Genuine IMDb labeled reviews; illustrative company pilot, not an independent final heldout validation',
          'csv_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.source.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    return output,meta

if __name__=='__main__':
    path,meta=prepare()
    print('REAL DATASET:',meta['dataset'],meta['split'],'|',meta['rows'],'original labeled reviews')
    print('FILE:',path)
    print('SOURCE FILE:',path.with_suffix('.source.json'))
    print('STATUS: REAL SOURCE DATA — NOT SYNTHETIC')
