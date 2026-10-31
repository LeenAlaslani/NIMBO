import unittest
from nimbo.guardrails_core import Policy, Observation, Evidence, Action, decide

class CoreTests(unittest.TestCase):
    def policy(self, kind='text_generation', **kw): return Policy(task_type=kind,max_latency_s=5,**kw)
    def obs(self, evidence=Evidence.FAIL, **kw): return Observation(evidence=evidence,elapsed_s=1,attempts_used=1,**kw)
    def test_unvalidated_pass_cannot_be_certified(self):
        self.assertNotEqual(decide(self.policy(),self.obs(Evidence.PASS,score=.99,verifier_id='clip',verifier_validated=False)).action,Action.ACCEPT)
    def test_validated_pass(self):
        self.assertEqual(decide(self.policy(),self.obs(Evidence.PASS,score=.96,verifier_id='fixture',verifier_validated=True)).action,Action.ACCEPT)
    def test_under_floor(self):
        self.assertEqual(decide(self.policy(),self.obs(Evidence.PASS,score=.8,verifier_id='fixture',verifier_validated=True)).action,Action.RETRY)
    def test_escalation(self):
        self.assertEqual(decide(self.policy('image_generation'),self.obs(escalation_available=True)).action,Action.ESCALATE)
    def test_exhaustion(self):
        self.assertEqual(decide(self.policy(),Observation(Evidence.FAIL,1,2)).action,Action.UNVERIFIED)
    def test_latency(self):
        self.assertEqual(decide(self.policy(),Observation(Evidence.FAIL,4.96,1)).action,Action.UNVERIFIED)
    def test_overbudget_pass(self):
        self.assertEqual(decide(self.policy(),Observation(Evidence.PASS,5.1,1,.99,'fixture',True)).action,Action.UNVERIFIED)
    def test_uncertain(self):
        self.assertEqual(decide(self.policy('text_classification'),self.obs(Evidence.UNCERTAIN)).action,Action.RETRY)
    def test_invalid(self):
        with self.assertRaises(ValueError): Policy('images',1)
        with self.assertRaises(ValueError): Observation(Evidence.PASS,-1,1)
        with self.assertRaises(ValueError): Observation(Evidence.PASS,1,1,float('nan'))
    def test_no_escalate_twice(self):
        self.assertEqual(decide(self.policy(),self.obs(escalation_available=True,already_escalated=True)).action,Action.RETRY)

if __name__=='__main__': unittest.main()
