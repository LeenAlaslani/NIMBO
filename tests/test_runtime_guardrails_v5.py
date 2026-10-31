import unittest, tempfile
from pathlib import Path
from unittest.mock import patch
from nimbo import runtime_guardrails_v5 as g
class TestRuntime(unittest.TestCase):
    def test_unknown_not_certified(self):
        a=g.evaluate('generation',100)
        self.assertEqual(a['action'],'UNVERIFIED')
        self.assertFalse(a['quality_certified'])
    def test_over_budget(self):
        a=g.evaluate('classification',6000)
        self.assertFalse(a['latency_within_budget'])
        self.assertEqual(a['action'],'UNVERIFIED')
    def test_unapproved_verifier(self):
        a=g.evaluate('image',100,verified=True,verifier_id='client_supplied')
        self.assertFalse(a['quality_certified'])
    def test_log_without_user_content(self):
        with tempfile.TemporaryDirectory() as td,patch.object(g,'LOG',Path(td)/'events.jsonl'):
            report={'guardrail':g.evaluate('generation',150),'metrics':{'end_to_end_ms':150},'result':{'answer':'PRIVATE DATA'}}
            g.record('generation','qwen',report)
            r=g.recent()
            self.assertEqual(len(r),1)
            self.assertNotIn('PRIVATE DATA',str(r))
if __name__=='__main__':unittest.main()
