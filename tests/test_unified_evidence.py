import json,tempfile,unittest
from pathlib import Path
from nimbo.unified_evidence import collect
class TestEvidence(unittest.TestCase):
 def test_types_do_not_mix(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   (p/'runtime_evidence_v5.jsonl').write_text(json.dumps({'workload':'classification','action':'UNVERIFIED','end_to_end_ms':22})+'\n')
   (p/'recovery_v8_oracle_bound.json').write_text(json.dumps({'sample_count':16,'successes':14,'generation_seconds_per_success':2.33}))
   result=collect(p)
   self.assertEqual(result['runtime_summary']['request_count'],1)
   self.assertEqual(len([r for r in result['records'] if r['evidence_kind']=='oracle_upper_bound']),1)
   self.assertIsNone(result['records'][0]['success_label'])
 def test_bad_jsonl_is_not_crash(self):
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'runtime_evidence_v5.jsonl').write_text('broken\n')
   self.assertEqual(collect(d)['runtime_summary']['invalid_lines'],1)
 def test_missing_optional_sources(self):
  with tempfile.TemporaryDirectory() as d:self.assertEqual(collect(d)['records'],[])
if __name__=='__main__':unittest.main()
