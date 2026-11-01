"""Dependency-light tests. Real Hugging Face inference requires an external Mac run."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from app.nimbo_model_intake_v1 import (check_model_id, inspect_hub_metadata,
    read_evaluation_file, resolve_labels, percentile, summarize_baseline)


def hub(**overrides):
    obj = dict(sha="a" * 40, siblings=[SimpleNamespace(rfilename="model.safetensors", size=142000)], cardData={"license":"apache-2.0"})
    obj.update(overrides)
    return SimpleNamespace(**obj)


def config(**overrides):
    obj = dict(model_type="distilbert", num_labels=2, id2label={0:"NEGATIVE",1:"POSITIVE"})
    obj.update(overrides)
    return SimpleNamespace(**obj)


@pytest.mark.parametrize("value", ["https://huggingface.co/a/b", "../repo", "a/../../b", "repo", "a/b?revision=main", "a/b/c", "/tmp/model", "a/.", "a/.."])
def test_reject_unsafe_ids(value):
    with pytest.raises(ValueError): check_model_id(value)


def test_accept_new_public_model_id_not_allowlisted():
    assert check_model_id("different-team/another-sentiment-classifier") == "different-team/another-sentiment-classifier"


def test_metadata_pins_revision_and_preserves_labels():
    value = inspect_hub_metadata("team/unseen-classifier", hub(), config())
    assert value.model_id == "team/unseen-classifier"
    assert value.resolved_revision == "a"*40 and value.id2label[1] == "POSITIVE"
    assert value.safe_weight_bytes == 142000 and value.license == "apache-2.0"


@pytest.mark.parametrize("h,c", [(hub(sha="main"), config()),(hub(siblings=[]),config()),
    (hub(siblings=[SimpleNamespace(rfilename="pytorch_model.bin",size=120)]),config()),
    (hub(siblings=[SimpleNamespace(rfilename="model.safetensors",size=999999999)]),config()),
    (hub(siblings=[SimpleNamespace(rfilename="model.safetensors",size=None)]),config()),
    (hub(),config(model_type="unknown")),(hub(),config(num_labels=0)),
    (hub(),config(id2label={0:"NEGATIVE"})),(hub(private=True),config())])
def test_fail_closed_unverifiable_model(h,c):
    with pytest.raises(ValueError): inspect_hub_metadata("team/model",h,c)


def test_real_csv_parsing_and_hash(tmp_path):
    p=tmp_path/"data.csv";p.write_text('text,label\n"hello, world",POSITIVE\n"bad",NEGATIVE\n',encoding="utf8")
    rows,digest=read_evaluation_file(p)
    assert len(rows)==2 and rows[0]["text"]=="hello, world" and len(digest)==64
    assert resolve_labels(rows, {0:"NEGATIVE",1:"POSITIVE"})==[1,0]


def test_jsonl_label_numbers(tmp_path):
    p=tmp_path/"rows.jsonl";p.write_text('\n'.join([json.dumps({"text":"first","label":0}),json.dumps({"text":"second","label":1})]),encoding="utf8")
    rows,_=read_evaluation_file(p)
    assert resolve_labels(rows,{0:"LABEL_0",1:"LABEL_1"})==[0,1]


def test_ambiguous_unknown_label_fails():
    with pytest.raises(ValueError,match="does not match"):
        resolve_labels([{"text":"x","label":"POSITIVE"}],{0:"LABEL_0",1:"LABEL_1"})


def test_unlabeled_not_accuracy_claim():
    data=summarize_baseline([0,1],[None,None],[10.0,20.0],[12.0,22.0])
    assert data["accuracy"] is None and data["accuracy_scope"]=="unverified_missing_labels"
    assert data["mean_warm_forward_ms"]==15.0


def test_correct_labeled_accuracy_and_p95():
    data=summarize_baseline([0,1,1],[0,1,0],[1.0,2.0,3.0],[2.0,3.0,4.0])
    assert data["accuracy"]==round(2/3,6) and data["p95_warm_forward_ms"]==2.9
    assert percentile([1,2],0.95)==1.95


def test_empty_malformed_csv_rejected(tmp_path):
    p=tmp_path/"x.csv";p.write_text("label\n0\n1\n")
    with pytest.raises(ValueError):read_evaluation_file(p)


def test_large_file_rejected_before_read(tmp_path):
    p=tmp_path/"x.csv";p.write_bytes(b"A" * (1024*1024+1))
    with pytest.raises(ValueError,match="1 MB"):read_evaluation_file(p)


def test_malformed_jsonl_rejected(tmp_path):
    p=tmp_path/"x.jsonl";p.write_text('{"text":"a"}\nnot-json\n')
    with pytest.raises(ValueError,match="line 2"):read_evaluation_file(p)
