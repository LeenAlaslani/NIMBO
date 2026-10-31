import unittest
from nimbo.context_answer_guard_v23 import answer_guard
E='Refund deadline is 14 days.\nRefund processing fee is $5.'
class QualityGuardTest(unittest.TestCase):
 def test_placeholder(self):self.assertEqual(answer_guard('deadline fee',E,'Deadline is [1], fee is [2]').status,'NEEDS_RETRY')
 def test_citations_not_numbers(self):self.assertEqual(answer_guard('deadline fee',E,'Deadline is 14 days, and fee is $5. [1][2]').status,'PENDING_HUMAN_QUALITY_REVIEW')
 def test_unseen_number(self):self.assertEqual(answer_guard('deadline',E,'Deadline is 30 days').status,'NEEDS_RETRY')
 def test_multihop(self):self.assertEqual(answer_guard('approver','Project Atlas is owned by Orion team.\nOrion team approvals go through Maya.','Maya approves Project Atlas via Orion. [1][2]').status,'PENDING_HUMAN_QUALITY_REVIEW')
 def test_abstain(self):self.assertEqual(answer_guard('warranty',E,'Not in context.').status,'NEEDS_RETRY')
 def test_missing_source(self):self.assertEqual(answer_guard('warranty',E,'Unknown','MISSING_EVIDENCE').status,'ABSTAIN_REQUIRED')
 def test_out_of_range_citation(self):self.assertEqual(answer_guard('deadline fee',E,'14 days and $5 [3]').status,'NEEDS_REVIEW')
if __name__=='__main__':unittest.main()
