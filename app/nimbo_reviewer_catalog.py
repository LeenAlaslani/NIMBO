"""A single fixed public workload for the hosted optimizer."""
MODEL = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
REVISION = '714eb0fa89d2f80546fda750413ed43d93601a13'
DATASET = 'stanfordnlp/sst2'
DATA_REVISION = '8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb'
CASE = 'distilbert-sst2'
PILOT = 24
HELDOUT = 200
RUNTIME = {'threads': 1, 'length': 128, 'rounds': 3}
REQUEST = {'model_id': MODEL, 'revision': REVISION, 'pilot_file': 'pilot.csv',
           'heldout_file': 'heldout.csv', 'preprocess': 'identity', 'objective': 'speed'}
