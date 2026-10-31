import unittest
from nimbo.live_guardrail_controller import check_answer,next_action,retry_guidance
Q='Who approves the release linked to Project Atlas?'
E='Project Atlas is owned by Orion team.\nOrion team approvals go through Maya.'
class Tests(unittest.TestCase):
 def test_intermediate(self):self.assertEqual(check_answer(Q,E,'Orion team approves it')[0],'RETRY')
 def test_person(self):self.assertEqual(check_answer(Q,E,'Maya approves it')[0],'RULE_SUPPORTED')
 def test_refund_unverified(self):self.assertEqual(check_answer('What is fee?','Fee is $5','Fee is $5')[0],'UNVERIFIED')
 def test_budget(self):self.assertEqual(next_action('RETRY',1,2,11,12,2),'STOP_BUDGET')
 def test_retry(self):self.assertEqual(next_action('RETRY',1,2,2,12,2),'RETRY')
 def test_max(self):self.assertEqual(next_action('RETRY',2,2,2,12,2),'STOP_ATTEMPTS')
 def test_guidance(self):self.assertIn('Maya',retry_guidance(Q,'Maya'))
if __name__=='__main__':unittest.main()
