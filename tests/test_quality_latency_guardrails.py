import unittest
from nimbo.quality_latency_guardrails import verify,linked_approver,decision
Q='Who approves the release linked to Project Atlas?'
C='Project Atlas is owned by Orion team.\nOrion team approvals go through Maya.'
class GuardrailTests(unittest.TestCase):
    def test_extract(self): self.assertEqual(linked_approver(Q,C),('Orion team','Maya'))
    def test_wrong_intermediate(self): self.assertEqual(verify(Q,C,'The Orion team approves it.').status,'RETRY')
    def test_right_approver(self): self.assertEqual(verify(Q,C,'Maya approves the release through Orion team.').status,'SUPPORTED_BY_RULE')
    def test_abstain_with_evidence(self): self.assertEqual(verify(Q,C,'Not in context.').status,'RETRY')
    def test_no_relation_does_not_certify(self): self.assertEqual(verify('What is the fee?','Fee is $5.','It costs $5.').status,'UNVERIFIED')
    def test_no_hallucinated_chain(self): self.assertEqual(verify(Q,'Project Atlas is owned by Orion team.','Orion approves.').status,'UNVERIFIED')
    def test_budget(self):
        self.assertEqual(decision(verify(Q,C,'Orion team'),1,2,0.1,1.0),'STOP_BUDGET')
        self.assertEqual(decision(verify(Q,C,'Orion team'),1,2,2,1),'RETRY')
        self.assertEqual(decision(verify(Q,C,'Orion team'),2,2,2,1),'STOP_UNVERIFIED')
if __name__=='__main__': unittest.main()
