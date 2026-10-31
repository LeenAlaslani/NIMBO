import unittest
from nimbo.context_optimizer_v2 import select_context_v2
class Tok:
    def encode(self,text,add_special_tokens=False):
        import re
        return re.findall(r"[\w\u0600-\u06ff]+|[^\w\s]",text)
class Tests(unittest.TestCase):
    def run_case(self,q,c,req=()):
        return select_context_v2(q,c,Tok(),80,req)
    def test_refund_distinct_facts(self):
        x=self.run_case('What is the refund deadline and fee?', 'Refund deadline is 14 days.\nRefund processing fee is $5.\n'+'Noise unrelated.\n'*40,('14 days','$5'))
        self.assertEqual(x.status,'OPTIMIZED')
        self.assertIn('14 days',x.context)
        self.assertIn('$5',x.context)
    def test_same_statement_conflict(self):
        x=self.run_case('When is the appointment on Friday?', 'Friday appointment is at 10:00.\nFriday appointment is at 11:00.\n'+'Noise unrelated.\n'*40)
        self.assertEqual(x.status,'NEEDS_REVIEW')
    def test_missing(self):
        x=self.run_case('What is the warranty period?', 'Refund deadline is 14 days.\n'+'Old unrelated.\n'*40,('warranty',))
        self.assertEqual(x.status,'MISSING_EVIDENCE')
    def test_multihop(self):
        x=self.run_case('Who approves the release linked to Project Atlas?', 'Project Atlas is owned by Orion team.\nOrion team approvals go through Maya.\n'+'Noise unrelated.\n'*40)
        self.assertEqual(x.status,'OPTIMIZED')
        self.assertIn('Maya',x.context)
    def test_unequal_numbers_different_subject(self):
        x=self.run_case('What is the shipping fee and delivery time?', 'Shipping fee is $7.\nDelivery time is 3 days.\n'+'Noise unrelated.\n'*40,('$7','3 days'))
        self.assertEqual(x.status,'OPTIMIZED')
if __name__=='__main__':unittest.main()
