import unittest
from nimbo.context_optimizer import select_context
class ToyTokenizer:
 def encode(self,s,add_special_tokens=False):return s.split()
class Tests(unittest.TestCase):
 def test_critical_evidence(self):
  r=select_context('refund deadline','office color blue. '*20+'Refund deadline 14 days. Fee $5.',ToyTokenizer(),15,['14 days','$5'])
  self.assertEqual(r.status,'optimized');self.assertIn('14 days',r.context);self.assertIn('$5',r.context)
 def test_too_small_keeps_original(self):
  r=select_context('deadline','More info. Refund deadline 14 days. Fee $5.',ToyTokenizer(),2,['14 days','$5'])
  self.assertNotEqual(r.status,'optimized');self.assertIn('14 days',r.context)
 def test_no_invented_facts(self):
  r=select_context('refund deadline','Deadline 14 days. A blue logo.',ToyTokenizer(),10,['$5'])
  self.assertEqual(r.status,'missing_required_phrase_in_source')
 def test_under_budget(self):
  r=select_context('x','tiny context',ToyTokenizer(),10)
  self.assertEqual(r.status,'unchanged_under_budget')
if __name__=='__main__':unittest.main()
