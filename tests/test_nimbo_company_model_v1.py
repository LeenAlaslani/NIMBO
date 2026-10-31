import asyncio
from pathlib import Path
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from app.nimbo_company_model_v1 import PilotRequest,Example,validate_rows,is_staged_and_verified,MAX_ROWS
from tools.prepare_real_pilot_dataset_v1 import prepare

def test_genuine_pilot_data_extracted_unchanged(tmp_path):
    ds=[{'text':'Genuine review body #'+str(i)+'. The plot and performances were memorable.'*(i%3+1),'label':i%2} for i in range(80)]
    csv,meta=prepare(ds,tmp_path/'test.csv')
    import csv as cs
    rows=list(cs.DictReader(csv.open()))
    assert len(rows)==12
    assert sum(x['label']=='POSITIVE' for x in rows)==6
    assert all(x['text'] in [v['text'] for v in ds] for x in rows)
    assert meta['dataset']=='stanfordnlp/imdb'

def test_labels_validate():
    p=PilotRequest(examples=[Example(text='great',expected_label='POSITIVE'),Example(text='bad',expected_label='NEGATIVE')])
    assert validate_rows(p.examples)==[('great',1),('bad',0)]

def test_incomplete_labels_are_rejected():
    p=PilotRequest(examples=[Example(text='a',expected_label='POSITIVE'),Example(text='b',expected_label='unknown')])
    with pytest.raises(ValueError):validate_rows(p.examples)

def test_min_two_examples_enforced():
    with pytest.raises(ValidationError):PilotRequest(examples=[Example(text='one',expected_label='POSITIVE')])

def test_max_rows_enforced():
    with pytest.raises(ValidationError):PilotRequest(examples=[Example(text='x',expected_label='POSITIVE')]*(MAX_ROWS+1))

def test_invalid_model_file_never_executed(tmp_path):
    file=tmp_path/'untrusted.onnx';file.write_bytes(b'fake')
    assert is_staged_and_verified(file) is False

def test_symlink_rejected(tmp_path):
    other=tmp_path/'real.onnx';other.write_bytes(b'x'*1024)
    link=tmp_path/'link.onnx';link.symlink_to(other)
    assert is_staged_and_verified(link) is False
