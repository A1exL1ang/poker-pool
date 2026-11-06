"""Check replay fidelity of saved development results after final policy changes."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import tournament
from tools.experiment_models import load_model
from tools.reference_policy import Agent as Stage1
from tools.refinement_policy import Agent as Stage2

checked=0
for filename,model,policy in [('tuning.json','equity_model_24.json',Stage1),
                               ('refinement.json','equity_model_8.json',Stage2)]:
    data=json.loads((Path('results')/filename).read_text())
    load_model(Path('assets')/model)
    for variant in data['results'].values():
        for i,old in enumerate(variant['rows'][:10]):
            new=tournament(old['seed'],variant['config'],'self' if i%5 else 'mixed',
                           'normal' if i%3 else 'slow' if i%2 else 'fast',policy_class=policy)
            for field in ('scores','stacks','hands','errors','internal_fallbacks'):
                assert new[field]==old[field],(filename,variant['config'],i,field,new[field],old[field])
            checked+=1
report={'tables_checked':checked,'result':'exact equality of scores/stacks/handcounts/errors/fallbacks',
        'note':'Ten seeds per saved variant. Timing intentionally excluded. Stage2 preserves prior observed-opportunity handler; final candidate contains the subsequent correction.'}
Path('results/reproduction.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
