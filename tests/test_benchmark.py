from pathlib import Path
import pytest
from nimbo.benchmark import load_rows, macro_f1, assess, evaluate


def test_macro_f1_perfect():
    assert macro_f1(['a','b'], ['a','b']) == 1.0


def test_benchmark_real_callback_interface():
    rows = [{'text':'x','label':'A'}, {'text':'y','label':'B'}]
    small = lambda s: {'label':'A' if s=='x' else 'B', 'confidence':.95}
    large = lambda s: {'label':'A' if s=='x' else 'B', 'confidence':.99}
    result = evaluate(rows,small,large,warmup=0)
    assert result['nimbo_adaptive']['accuracy'] == 1.0
    assert result['nimbo_adaptive']['routes']['small-first'] == 2


def test_fallback_is_counted():
    rows = [{'text':'x','label':'A'}]
    result = evaluate(rows, lambda s:{'label':'B','confidence':.2}, lambda s:{'label':'A','confidence':.99},warmup=0)
    assert result['nimbo_adaptive']['routes']['small-then-large'] == 1


def test_reject_bad_csv(tmp_path: Path):
    f=tmp_path/'bad.csv'; f.write_text('description,type\nhi,a\n')
    with pytest.raises(ValueError): load_rows(str(f))
