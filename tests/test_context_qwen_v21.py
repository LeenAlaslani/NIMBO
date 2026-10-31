import unittest
from experiments.context_qwen_v21_benchmark import choose_context,build_tasks
class TestBenchmark(unittest.TestCase):
 def test_four_cases(self): self.assertEqual(len(build_tasks()),4)
 def test_optimize(self):
  from types import SimpleNamespace
  self.assertEqual(choose_context(SimpleNamespace(status="OPTIMIZED",context="short"),"long"),("short","OPTIMIZED"))
 def test_conflict_abstains(self):
  from types import SimpleNamespace
  self.assertEqual(choose_context(SimpleNamespace(status="NEEDS_REVIEW",context="long"),"long"),("","ABSTAIN_NEEDS_REVIEW"))
 def test_budget_preserves_original(self):
  from types import SimpleNamespace
  self.assertEqual(choose_context(SimpleNamespace(status="BUDGET_UNSAFE",context="long"),"long"),("long","FULL_CONTEXT_BUDGET_UNSAFE"))
if __name__=="__main__":unittest.main()
