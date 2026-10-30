from experiments.benchmark_v2 import choose_threshold,run_real_benchmark

def test_threshold_falls_back_when_needed():
    s=[{'label':'A','confidence':.6},{'label':'A','confidence':.98}]
    l=[{'label':'B','confidence':.9},{'label':'A','confidence':.9}]
    assert choose_threshold(s,l,['B','A'],max_accuracy_drop=0) > .6

def test_real_pipeline_with_callbacks():
    rows=[{'text':'x','label':'POSITIVE'},{'text':'y','label':'NEGATIVE'}]
    def small(t): return {'label':'POSITIVE','confidence':.51 if t=='y' else .99}
    def large(t): return {'label':'POSITIVE' if t=='x' else 'NEGATIVE','confidence':.99}
    result=run_real_benchmark(rows,rows,small,large,max_drop=0,warmup=0)
    assert result['strategies']['adaptive']['accuracy']==1
    assert result['strategies']['small']['accuracy']==.5
    assert result['strategies']['adaptive']['fallback_rate']>0
