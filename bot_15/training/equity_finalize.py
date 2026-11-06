"""Export the fixed validation-selected tree prefix; no fitting or test access."""
import argparse
import gzip
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--source',default='artifacts/equity_model_full24.json.gz');p.add_argument('--out',default='artifacts/equity_model.json.gz');p.add_argument('--trees',type=int,default=8);args=p.parse_args()
with gzip.open(args.source,'rt') as f:model=json.load(f)
assert 0<args.trees<=len(model['trees'])
model['trees']=model['trees'][:args.trees]
model['runtime_selection']='Fixed tree prefix selected on validation accuracy and latency, not held-out test labels.'
temporary=Path(args.out+'.tmp')
with gzip.open(temporary,'wt',compresslevel=9) as f:json.dump(model,f,separators=(',',':'))
temporary.replace(args.out)
print({'trees':args.trees,'bytes':Path(args.out).stat().st_size})
