import unittest
from nimbo.context_optimizer_v2 import select_context_v2
class Tok:
 def encode(self,s,add_special_tokens=False):return s.split()
class TestV2(unittest.TestCase):
 def test_missing(self):
  s=select_context_v2('warranty','unrelated notes '*90,Tok(),20,('warranty',));self.assertEqual(s.status,'MISSING_EVIDENCE')
 def test_conflict(self):
  s=select_context_v2('Friday appointment','Friday appointment 10.\nFriday appointment 11.\n'+'other filler '*70,Tok(),40);self.assertEqual(s.status,'NEEDS_REVIEW')
 def test_multihop(self):
  s=select_context_v2('Project Atlas owner','Project Atlas owned by Orion.\nOrion approvals are Maya.\n'+'irrelevant filler '*90,Tok(),40);self.assertEqual(s.status,'OPTIMIZED');self.assertIn('Maya',s.context)
 def test_required(self):
  s=select_context_v2('refund fee','Refund fee $5.\n'+'unrelated filler '*80,Tok(),20,('$5',));self.assertIn('$5',s.context)
 def test_fail_closed(self):
  s=select_context_v2('refund fee','Refund fee has a very long critical condition.\n'+'filler '*80,Tok(),1,('critical condition',));self.assertEqual(s.status,'BUDGET_UNSAFE')
if __name__=='__main__':unittest.main()
