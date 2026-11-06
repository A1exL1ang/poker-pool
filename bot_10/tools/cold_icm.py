"""Time the very first ICM decision without preimporting the simulator/harness."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
t=time.perf_counter_ns()
from agent import Agent
loaded=time.perf_counter_ns()
assert 'icm' in sys.modules
a=Agent(10010)
o=dict(hole=[48,49],board=[],hero=0,button=7,small_blind=10,big_blind=20,pot=120,
       to_call=80,min_raise_to=180,max_raise_to=100,can_raise=False,street=0,hand_id=1,
       payouts=[.5,.3,.2,0,0,0,0,0],
       players=[dict(seat=j,stack=80 if j==0 else 0 if j==1 else 1000,
                     bet=20 if j==0 else 100 if j==1 else 0,
                     committed=20 if j==0 else 100 if j==1 else 0,
                     folded=j>1,all_in=j==1) for j in range(8)],
       history=[dict(seat=1,street=0,action='raise',amount=100)])
start=time.perf_counter_ns();action=a.act(o);duration=time.perf_counter_ns()-start
assert action['action']=='call' and not a.failures
report={'import_ms':(loaded-t)/1e6,'first_icm_act_us':duration/1000,
        'icm_loaded_before_decision':True,'action':action,'python':sys.version,
        'note':'One local Windows wall-clock observation. Import time excluded from act; no simulator/benchmark imported before timing.'}
Path('results/cold_icm.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
