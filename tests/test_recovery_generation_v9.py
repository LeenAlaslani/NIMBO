import unittest
from nimbo.recovery_generation_v9 import run,repair_prompt
class Tests(unittest.TestCase):
    def test_fast_pass(self):
        r=run('a dog',lambda s,p,i:s,lambda im,p:('PASS','seen'))
        self.assertEqual(r.status,'CHECKER_PASS_UNVALIDATED');self.assertEqual(len(r.attempts),1)
    def test_retry_with_repair_and_new_index(self):
        records=[]
        def gen(s,p,i):records.append((s,p,i));return s
        r=run('an astronaut holding blue flag',gen,
              lambda im,p:('PASS','yes') if im=='repaired_standard' else ('FAIL','flag missing'),critical_object='blue flag')
        self.assertEqual(len(records),3);self.assertIn('blue flag',records[-1][1]);self.assertEqual(records[-1][2],2)
        self.assertTrue(r.checker_pass)
    def test_all_fail(self):
        r=run('p',lambda s,p,i:s,lambda im,p:('FAIL','not found'))
        self.assertEqual(r.status,'FAILED_NO_DELIVERY');self.assertIsNone(r.image)
    def test_uncertain_not_pass(self):
        r=run('p',lambda s,p,i:s,lambda im,p:('UNCERTAIN','unclear'))
        self.assertFalse(r.checker_pass)
    def test_checker_error_safe(self):
        def err(*args):raise RuntimeError('bad')
        r=run('p',lambda s,p,i:s,err)
        self.assertEqual(r.status,'FAILED_NO_DELIVERY')
    def test_budget_no_delivery(self):
        ticks=iter([0,0,0,0,2,2])
        r=run('p',lambda s,p,i:s,lambda im,p:('PASS','ok'),clock=lambda:next(ticks),budget_s=1)
        self.assertFalse(r.checker_pass)
    def test_repair(self):
        self.assertIn('number of objects',repair_prompt('single sunflower'))
if __name__=='__main__': unittest.main()
