from nimbo.core import NimboAgent, Request, Modality

def get_agent(confidence=.9):
    return NimboAgent(lambda x: {'label':'A','confidence':confidence},
                      lambda x: {'label':'B','confidence':.95},
                      lambda x: 'fast_image', lambda x: 'quality_image')

def test_small_route():
    t=get_agent().run(Request(Modality.TEXT,'hello'))
    assert t.decision['route'] == 'small-first' and t.result['label']=='A'

def test_escalation():
    t=get_agent(.2).run(Request(Modality.TEXT,'hello'))
    assert t.decision['route'] == 'small-then-large' and t.result['label']=='B'
    assert any(x['state']=='escalate' for x in t.events)

def test_image_route():
    assert get_agent().run(Request(Modality.IMAGE,'city','preview')).result=='fast_image'
    assert get_agent().run(Request(Modality.IMAGE,'city','high')).result=='quality_image'

def test_no_fake_cost():
    assert get_agent().run(Request(Modality.TEXT,'hi')).estimated_cost_usd is None
