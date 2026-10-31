import unittest
from nimbo.context_answer_guard_v22 import check_answer,evidence_instructions
class TestGuard(unittest.TestCase):
    def test_unknown_number_rejected(self):
        self.assertEqual(check_answer('When?', 'Deadline is 14 days.', 'Deadline is 30 days.', 'OPTIMIZED').status,'NEEDS_REVIEW')
    def test_abstention_flagged(self):
        self.assertEqual(check_answer('Who?', 'Orion approvals go through Maya.', 'Not in context.', 'OPTIMIZED').status,'NEEDS_REVIEW')
    def test_not_automatic_quality_pass(self):
        self.assertEqual(check_answer('Who?', 'Orion approvals go through Maya.', 'Maya', 'OPTIMIZED').status,'PENDING_HUMAN_QUALITY_REVIEW')
    def test_missing_status_blocks(self):
        self.assertEqual(check_answer('Who?', 'text', 'Maya', 'MISSING_EVIDENCE').status,'ABSTAIN_REQUIRED')
    def test_relational_instruction(self):
        self.assertIn('connect facts from different lines',evidence_instructions('Who?', 'A belongs to B.'))
if __name__=='__main__':unittest.main()
