import json
import pytest
from tools.verify_cross_domain_public_v1 import PATH, verify

def test_cross_domain_published_evidence_is_consistent():
    assert verify(json.loads(PATH.read_text(encoding='utf8')))

def test_cross_domain_tampered_cpu_reduction_fails():
    d=json.loads(PATH.read_text(encoding='utf8'))
    d['process_cpu_time_reduction_fraction']=-0.5
    with pytest.raises(ValueError,match='cpu arithmetic'):verify(d)

def test_cross_domain_no_fake_deployment():
    d=json.loads(PATH.read_text(encoding='utf8'))
    d['production_deployed']=True
    with pytest.raises(ValueError,match='decision/deployment'):verify(d)
