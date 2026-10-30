import pytest
from nimbo.text_backend import normalize_sst2_label

def test_normalize_model_labels():
    assert normalize_sst2_label("LABEL_0") == "NEGATIVE"
    assert normalize_sst2_label("LABEL_1") == "POSITIVE"
    assert normalize_sst2_label("positive") == "POSITIVE"

def test_unknown_label_rejected():
    with pytest.raises(ValueError): normalize_sst2_label("neutral")
