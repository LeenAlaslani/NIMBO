from experiments.evaluate_image_review_v2 import summarize

def row(accepted,fast,standard):
    return {'route':'fast_accepted' if accepted else 'escalated_to_standard',
            'human_fast_object_present':str(int(fast)),
            'human_standard_object_present':str(int(standard))}

def test_all_four_outcomes():
    r=summarize([row(True,True,True),row(False,False,True),row(True,False,True),row(False,True,False)])
    assert r['confusion']=={'correct_accept':1,'correct_escalate':1,'unsafe_accept':1,'unnecessary_escalate':1}
    assert r['checker_accuracy']==.5
    assert r['human_object_success_rate']['nimbo']==.5

def test_missing_labels_ignored():
    r=summarize([row(True,True,True),{'route':'fast_accepted','human_fast_object_present':'','human_standard_object_present':'0'}])
    assert r['labeled']==1
    assert r['checker_accuracy']==1

def test_no_labels():
    assert summarize([{'route':'fast_accepted'}])['labeled']==0
