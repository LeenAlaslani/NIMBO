from tools.external_domain_v1 import (
    pick_preregistered,row_digest,summary,reduction,assess_cross_domain,
    DATASET_REVISION,DATASET_PARQUET_PATH,load_pinned_tweet_eval_test,verify_review,SOURCE_RUN,
)
import pytest

def sample_rows():
    return ([{'text':f'bad-{i}','label':0} for i in range(30)] +
            [{'text':f'neutral-{i}','label':1} for i in range(30)] +
            [{'text':f'good-{i}','label':2} for i in range(30)])

def test_fixed_selection_is_balanced_and_deterministic():
    a=pick_preregistered(sample_rows(),per_class=10)
    b=pick_preregistered(sample_rows(),per_class=10)
    assert a==b and len(a)==20 and len({i for i,_,_ in a})==20
    assert sum(y==0 for _,_,y in a)==10 and sum(y==1 for _,_,y in a)==10
    assert not any('neutral-' in t for _,t,_ in a)
    assert len(row_digest(a))==64 and row_digest(a)==row_digest(b)

def test_changes_in_labels_are_in_digest():
    a=pick_preregistered(sample_rows(),per_class=5)
    changed=list(a);i,t,y=changed[0];changed[0]=(i,t,1-y)
    assert row_digest(a)!=row_digest(changed)

@pytest.mark.parametrize('bad', [[], [{'label':0,'text':''}], [{'label':3,'text':'x'}]])
def test_invalid_rows_fail_closed(bad):
    with pytest.raises(ValueError):pick_preregistered(bad,per_class=1)

def record(wall,cpu,accuracy=.86,p95=None):
    return {'wall':{'mean_ms':wall,'p95_ms':p95 or wall*1.2},
            'cpu':{'mean_ms':cpu},'accuracy':accuracy}

def rounds(bw=20,cw=17,bc=40,cc=38):
    return [{'baseline_wall_mean_ms':bw,'candidate_wall_mean_ms':cw,
             'baseline_cpu_mean_ms':bc,'candidate_cpu_mean_ms':cc} for _ in range(3)]

def test_cross_domain_good_candidate_passes():
    d=assess_cross_domain(record(20,40),record(17,38),agreement=1,rounds=rounds())
    assert d['all_cross_domain_review_gates_pass'] and d['absolute_domain_accuracy_75pct_diagnostic']

def test_cross_domain_rejects_cpu_increase_even_when_faster():
    d=assess_cross_domain(record(20,40),record(16,50),agreement=1,rounds=rounds(bc=40,cc=50,cw=16))
    assert not d['cpu_gate_pass'] and not d['all_cross_domain_review_gates_pass']

def test_cross_domain_domain_shift_fails_absolute_accuracy_floor():
    d=assess_cross_domain(record(20,40,.65),record(17,38,.65),agreement=1,rounds=rounds())
    assert not d['absolute_domain_accuracy_75pct_diagnostic']
    assert not d['all_cross_domain_review_gates_pass']

def test_cross_domain_rejects_label_difference():
    d=assess_cross_domain(record(20,40),record(17,38,.83),agreement=.96,rounds=rounds())
    assert not d['prediction_parity_pass'] and not d['all_cross_domain_review_gates_pass']

def test_cross_domain_rejects_p95_regression():
    d=assess_cross_domain(record(20,40,p95=23),record(17,38,p95=27),agreement=1,rounds=rounds())
    assert not d['p95_pass'] and not d['all_cross_domain_review_gates_pass']

def test_missing_or_nonfinite_rounds_are_rejected():
    missing=assess_cross_domain(record(20,40),record(17,38),agreement=1,rounds=[])
    assert not missing['all_cross_domain_review_gates_pass']
    invalid=rounds(); invalid[1]['candidate_cpu_mean_ms']=float('nan')
    bad=assess_cross_domain(record(20,40),record(17,38),agreement=1,rounds=invalid)
    assert not bad['all_cross_domain_review_gates_pass']

def test_bad_inputs_rejected():
    with pytest.raises(ValueError): summary([])
    with pytest.raises(ValueError): reduction(0,10)
    with pytest.raises(ValueError): reduction(10,float('nan'))

def test_locked_source_checks_both_objective_and_threads():
    good={'run_id':SOURCE_RUN,'sha256':'a'*64,
          'manifest':{'candidate_cpu_threads':2,'cpu_threads':4,'max_length':256},
          'validation':{'dataset_fingerprint':'f','optimization':{'gate':{
              'optimization_objective':'cpu_time','eligible':True,'cpu_pass':True}}}}
    assert verify_review(good)
    bad={**good,'manifest':{**good['manifest'],'candidate_cpu_threads':4}}
    with pytest.raises(RuntimeError):verify_review(bad)

def test_pin_is_full_commit():
    assert len(DATASET_REVISION)==40 and all(c in '0123456789abcdef' for c in DATASET_REVISION)


def test_pinned_parquet_loader_uses_official_immutable_test_file(tmp_path):
    import hashlib
    content=b'fixed parquet test bytes'
    f=tmp_path/'0000.parquet';f.write_bytes(content)
    called={}
    def download(**kwargs):
        called['download']=kwargs
        return str(f)
    class Dataset:
        column_names=['text','label']
        def __len__(self):return 1000
    def load(*args,**kwargs):
        called['load']=(args,kwargs)
        return Dataset()
    ds,digest=load_pinned_tweet_eval_test(downloader=download,parquet_loader=load)
    assert len(ds)==1000
    assert digest==hashlib.sha256(content).hexdigest()
    assert called['download']=={'repo_id':'cardiffnlp/tweet_eval','repo_type':'dataset',
        'revision':DATASET_REVISION,'filename':DATASET_PARQUET_PATH}
    assert called['load'][0]==('parquet',)
    assert called['load'][1]=={'data_files':{'test':str(f)},'split':'test'}


def test_pinned_parquet_loader_fails_for_missing_file(tmp_path):
    def download(**kwargs):return str(tmp_path/'missing.parquet')
    with pytest.raises(RuntimeError,match='missing or empty'):
        load_pinned_tweet_eval_test(downloader=download,parquet_loader=lambda *a,**kw:None)


def test_pinned_parquet_loader_fails_for_bad_schema(tmp_path):
    f=tmp_path/'0000.parquet';f.write_bytes(b'not actually decoded')
    class Dataset:
        column_names=['tweet','sentiment']
        def __len__(self):return 1000
    with pytest.raises(RuntimeError,match='text and label'):
        load_pinned_tweet_eval_test(downloader=lambda **kw:str(f),parquet_loader=lambda *a,**kw:Dataset())
