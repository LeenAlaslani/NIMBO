from experiments.benchmark_v4 import select_policy,evaluate


def test_no_empty_routing_claim():
    rows=[{'text':'a','label':'POSITIVE'},{'text':'b','label':'NEGATIVE'}]
    small=[{'label':'POSITIVE','confidence':.99},{'label':'NEGATIVE','confidence':.99}]
    large=[{'label':'POSITIVE','confidence':.99},{'label':'NEGATIVE','confidence':.99}]
    policy,_,_=select_policy(rows,small,large)
    assert policy['type']=='small_only'


def test_adaptive_selected_only_when_useful():
    rows=[{'text':'a','label':'POSITIVE'},{'text':'b','label':'NEGATIVE'}]
    small=[{'label':'POSITIVE','confidence':.99},{'label':'POSITIVE','confidence':.60}]
    large=[{'label':'POSITIVE','confidence':.99},{'label':'NEGATIVE','confidence':.99}]
    policy,_,_=select_policy(rows,small,large,max_fallback=0.5)
    assert policy['type']=='adaptive'
    assert policy['threshold']==0.7


def test_pipeline_with_tiny_fast_callbacks():
    rows=[{'text':'a','label':'POSITIVE'},{'text':'b','label':'NEGATIVE'}]
    def small(text):
        return {'label':'POSITIVE','confidence':.6 if text=='b' else .99}
    def large(text):
        return {'label':'POSITIVE' if text=='a' else 'NEGATIVE','confidence':.99}
    result=evaluate(rows,small,large,{'type':'adaptive','threshold':.7},warmup=0)
    assert result['nimbo']['accuracy']==1.0
    assert result['nimbo']['fallback_rate']==.5
