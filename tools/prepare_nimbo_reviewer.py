"""Fetch the pinned public input snapshot when building the reviewer image."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
import pyarrow.parquet as pq

from app.nimbo_reviewer_catalog import DATASET, DATA_REVISION, HELDOUT, MODEL, PILOT, REVISION

MODEL_FILES = ['config.json', 'model.safetensors', 'tokenizer_config.json', 'vocab.txt']


def sample_rows(path, split, count, excluded):
    rows = pq.read_table(path, columns=['sentence', 'label']).to_pylist()
    chosen = []
    for label in (0, 1):
        candidates = sorted({row['sentence'].strip() for row in rows if row['label'] == label},
                            key=lambda sentence: hashlib.sha256(('nimbo-v1:' + sentence).encode()).digest())
        before = len(chosen)
        for sentence in candidates:
            key = sentence.casefold()
            if 1 <= len(sentence) <= 2000 and key not in excluded:
                chosen.append((sentence, label))
                excluded.add(key)
            if len(chosen) - before == count // 2:
                break
        if len(chosen) - before != count // 2:
            raise ValueError(f'Insufficient unique {split} rows for label {label}')
    return chosen


def prepare(directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / 'hub'
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=MODEL_FILES, cache_dir=cache))
    if any(not (snapshot / name).is_file() for name in MODEL_FILES):
        raise ValueError('The pinned model snapshot is incomplete')
    (directory / 'model.json').write_text(json.dumps({
        'model_id': MODEL, 'revision': REVISION, 'license': 'apache-2.0', 'snapshot': str(snapshot),
    }))
    excluded = set()
    for split, filename, count in [('train', 'pilot.csv', PILOT), ('validation', 'heldout.csv', HELDOUT)]:
        parquet = hf_hub_download(DATASET, f'data/{split}-00000-of-00001.parquet',
                                  repo_type='dataset', revision=DATA_REVISION)
        rows = sample_rows(parquet, split, count, excluded)
        output = directory / filename
        with output.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['text', 'label'])
            writer.writerows(rows)
        output.with_suffix('.source.json').write_text(json.dumps({
            'source': DATASET, 'revision': DATA_REVISION, 'split': split, 'rows': count,
            'csv_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        }))
        print(split, count, hashlib.sha256(output.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
