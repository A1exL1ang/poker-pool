"""Predeclared held-out scenarios. Run only after freezing the candidate."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from tools.benchmark import one, summarize

SCENARIOS = {
    'mixed':(1190000,1500,'mixed','none','double'),
    'tight':(1193000,1500,'tight','none','double'),
    'loose':(1196000,1500,'loose','none','double'),
    'selfplay':(1200000,500,'selfplay','none','double'),
    'synthetic_top3':(1202000,1000,'mixed','top3','double'),
    'synthetic_linear':(1204000,500,'mixed','linear','double'),
    'slower_blinds':(1206000,750,'mixed','none','1.5x'),
}


def tagged(task):
    name, inner = task
    return name,one(inner)


def main():
    paths = [Path('pokerbot')/name for name in ('agent.py','equity.py','features.py','evaluator.py','icm.py')]+[Path('assets/equity.bin.z'),Path('assets/preflop.json')]
    hashes = {p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows = {name:[] for name in SCENARIOS}
    tasks = []
    for name,(seed,count,eco,payout,blind) in SCENARIOS.items():
        tasks += [(name,(seed+i,eco,{},payout,False,blind)) for i in range(count)]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for name,row in pool.map(tagged,tasks,chunksize=8):
            rows[name].append(row)
    reports = {}
    for name,result in rows.items():
        report = {'heldout':True,'scenario':SCENARIOS[name],'tables':len(result),
            'chip_share':summarize(result,'chip_share'),'rank':summarize(result,'rank'),
            'totals':{k:sum(r[k] for r in result) for k in ('engine_errors','invalid_actions','agent_errors','hands')},
            'rows':result,'source_hashes':hashes}
        if result[0]['points'] is not None:
            report['synthetic_points'] = summarize(result,'points')
        Path(f'benchmarks/heldout_{name}.json').write_text(json.dumps(report,indent=2)+'\n')
        reports[name] = {k:v for k,v in report.items() if k not in ('rows','source_hashes')}
    changed = [p.as_posix() for p in paths if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[p.as_posix()]]
    if changed:
        raise RuntimeError(f'candidate changed during heldout evaluation: {changed}')
    summary = {'heldout':True,'seconds':time.monotonic()-started,'source_hashes':hashes,
               'source_changes_during_run':changed,'total_tables':len(tasks),'scenarios':reports,
               'latency_enforcement':False,'note':'Parallel performance tests do not measure isolated decision latency. Synthetic payout vectors and later blinds are local assumptions.'}
    Path('benchmarks/heldout_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
