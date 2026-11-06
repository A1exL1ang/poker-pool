"""Small, predeclared policy sensitivity study on DEVELOPMENT seeds only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import statistics
import time
from tools.benchmark import one, summarize

CONFIGS = {
    'default': {},
    'tight': {'open_scale': .88, 'call_margin': .025, 'value_threshold': .64, 'bluff_scale': .7},
    'loose': {'open_scale': 1.12, 'call_margin': .005, 'value_threshold': .60, 'bluff_scale': .8},
    'range_less': {'range_weight': .65, 'call_margin': .005},
    'range_more': {'range_weight': 1.2, 'call_margin': .02},
    'no_bluff': {'bluff_scale': 0., 'value_threshold': .60},
}


def tagged(task):
    name, ecosystem, seed = task
    return name, ecosystem, one((seed, ecosystem, CONFIGS[name], 'none', False, 'double'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tables', type=int, default=400, help='per config/ecosystem')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', default='benchmarks/policy_tuning.json')
    args = p.parse_args()
    starts = {'mixed':1105000, 'tight':1115000, 'loose':1125000}
    tasks = [(name, eco, seed+i) for name in CONFIGS for eco, seed in starts.items() for i in range(args.tables)]
    began = time.monotonic()
    rows = {name:{eco:[] for eco in starts} for name in CONFIGS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, eco, row in pool.map(tagged, tasks, chunksize=8):
            rows[name][eco].append(row)
    report = {'development_only':True, 'configurations':CONFIGS, 'tables_per_config_ecosystem':args.tables,
              'seed_starts':starts, 'seconds':time.monotonic()-began, 'results':{}, 'rows':rows}
    for name in CONFIGS:
        report['results'][name] = {}
        all_diff = []
        for eco in starts:
            candidate = sorted(rows[name][eco], key=lambda r:r['seed'])
            baseline = sorted(rows['default'][eco], key=lambda r:r['seed'])
            diff = [a['chip_share']-b['chip_share'] for a,b in zip(candidate,baseline)]
            all_diff += diff
            report['results'][name][eco] = {'chip_share':summarize(candidate,'chip_share'), 'rank':summarize(candidate,'rank'),
                'paired_difference_from_default':{'mean':statistics.mean(diff),'stderr':statistics.stdev(diff)/math.sqrt(len(diff))},
                'totals':{k:sum(r[k] for r in candidate) for k in ('engine_errors','invalid_actions','agent_errors')}}
        report['results'][name]['pooled_paired_difference'] = {'mean':statistics.mean(all_diff),'stderr':statistics.stdev(all_diff)/math.sqrt(len(all_diff))}
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__ == '__main__':
    main()
