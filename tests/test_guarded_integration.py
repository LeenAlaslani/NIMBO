import json, os, tempfile, unittest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
import sys, types
# Self-contained fake mission API ensures we don't assume the real app shape.
fake=types.ModuleType('app.mission_api')
fake.app=FastAPI()
@fake.app.get('/ping')
def ping(): return {'ok':True}
sys.modules['app.mission_api']=fake
from app import guarded_entry as g
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        g.log_path=Path(self.tmp.name)/'events.jsonl'
        self.client=TestClient(g.app)
    def tearDown(self):self.tmp.cleanup()
    def test_real_request_traced_not_certified(self):
        r=self.client.get('/ping');self.assertEqual(r.status_code,200)
        self.assertEqual(r.headers['X-NIMBO-Quality'],'NOT_VERIFIED')
        ev=self.client.get('/nimbo/evidence/recent').json()['events'][-1]
        self.assertEqual(ev['path'],'/ping');self.assertFalse(ev['quality_certified'])
    def test_untrusted_verifier_never_certifies(self):
        r=self.client.post('/nimbo/guardrails/decision',json={'policy':{'task_type':'text_generation','max_latency_s':4.0},'observation':{'evidence':'PASS','elapsed_s':.1,'attempts_used':1,'score':.99,'verifier_id':'user_claim','verifier_validated':True}})
        self.assertEqual(r.status_code,200);self.assertFalse(r.json()['decision']['quality_certified'])
        self.assertNotEqual(r.json()['decision']['action'],'ACCEPT')
    def test_no_text_persisted(self):
        self.client.get('/ping?token=topsecret')
        self.assertNotIn('topsecret',g.log_path.read_text())
if __name__=='__main__': unittest.main()
