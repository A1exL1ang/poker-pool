"""Predeclared held-out suites. Run only after policy/model freeze.

Results are descriptive evidence, never input to tuning. Parallel tournaments
measure actual wall limits too; dedicated latency runs afterwards in one process.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark import run_benchmark

SUITES = [
    ('mixed_sng', 'mixed', 'sng', 100000, 1000, True, None),
    ('mixed_wta', 'mixed', 'wta', 110000, 1000, True, None),
    ('reference_sng', 'reference', 'sng', 120000, 1000, True, None),
    ('mirror_sng', 'mirror', 'sng', 130000, 1000, True, None),
    ('mixed_flat3', 'mixed', 'flat3', 140000, 1000, True, None),
    ('mixed_fast_blinds', 'mixed', 'sng', 150000, 1000, True,
     [(10,20),(20,40),(40,80),(80,160),(160,320),(320,640),(640,1280),(1280,2560)]),
    ('mc_mixed_sng', 'mc,value,nit,loose,station,mirror,value', 'sng', 160000, 256, False, None),
    ('mc_sng', 'mc', 'sng', 170000, 64, False, None),
]


def task(suite):
    label,lineup,payout,seed,tables,enforce,blinds=suite
    args=SimpleNamespace(tables=tables,max_hands=200,seed_start=seed,split='heldout',
        lineup=lineup,params=None,blinds=json.dumps(blinds) if blinds else None,
        payouts_json=None,payouts=payout,model_path=None,initial_stack=2000,
        enforce_time_limit=enforce,no_timing=False,progress=0,stress=False)
    r=run_benchmark(args)
    Path('measurements').mkdir(exist_ok=True)
    Path(f'measurements/final_{label}.json').write_text(json.dumps(r,indent=2))
    return label,r


if __name__=='__main__':
    started=time.time()
    results={}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for label,r in pool.map(task,SUITES):
            results[label]={k:r[k] for k in ('config','focal_points','equal_share_reference',
                'finish_distribution','focal_diagnostics','all_diagnostics','focal_latency',
                'wall_seconds','artifact_sha256','runtime')}
            print(label,json.dumps(results[label]['focal_points']),flush=True)
    report={'purpose':'Held-out evidence after frozen policy; no subsequent tuning',
            'suite_definitions':SUITES,'elapsed_seconds':time.time()-started,'results':results}
    Path('measurements/final_evaluation.json').write_text(json.dumps(report,indent=2))
    print('BOT_9_FINAL_EVALUATION_DONE',flush=True)
