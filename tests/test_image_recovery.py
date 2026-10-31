import unittest
from nimbo.image_recovery import Check, Verdict, recover

class TestRecovery(unittest.TestCase):
    def test_fast_pass(self):
        r=recover('prompt',lambda stage,i:stage,lambda im,p:Check(Verdict.PASS),budget_seconds=10)
        self.assertTrue(r.verified)
        self.assertEqual([x.stage for x in r.attempts],['fast'])
    def test_escalation(self):
        r=recover('prompt',lambda stage,i:stage,lambda im,p:Check(Verdict.FAIL if im=='fast' else Verdict.PASS),budget_seconds=10)
        self.assertTrue(r.verified)
        self.assertEqual(r.image,'standard')
    def test_both_fail_never_deliver(self):
        r=recover('prompt',lambda stage,i:stage,lambda im,p:Check(Verdict.FAIL),budget_seconds=10)
        self.assertEqual(r.status,'FAILED');self.assertIsNone(r.image)
    def test_uncertain_never_pass(self):
        r=recover('prompt',lambda stage,i:stage,lambda im,p:Check(Verdict.UNCERTAIN),budget_seconds=10)
        self.assertFalse(r.verified)
    def test_verification_exception(self):
        def bad(*_): raise ValueError('bad')
        r=recover('prompt',lambda stage,i:stage,bad,budget_seconds=10)
        self.assertFalse(r.verified)
    def test_budget_before_first(self):
        ticks=iter([0.,10.,10.]); r=recover('p',lambda s,i:s,lambda im,p:Check(Verdict.PASS),budget_seconds=1,clock=lambda:next(ticks))
        self.assertEqual(r.status,'BUDGET_EXHAUSTED')
    def test_validation(self):
        with self.assertRaises(ValueError):recover('p',lambda s,i:s,lambda im,p:Check(Verdict.PASS),max_attempts=3)
if __name__=='__main__': unittest.main()
