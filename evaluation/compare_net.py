"""Compare frozen pure-net and PPO experiments with seed-block uncertainty."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze import summarize

ROOT=Path(__file__).resolve().parent

def read(folder):
    return [json.loads(line) for p in sorted(folder.glob('production_*.jsonl')) for line in p.read_text().splitlines()]

def main():
    net=read(ROOT/'net_results');ppo=read(ROOT/'results')
    nsummary=json.loads((ROOT/'net_summary.json').read_text())
    psummary=json.loads((ROOT/'summary.json').read_text())
    assert len(net)==len(ppo)==13824
    assert all(r['baseline_variant']=='net' for r in net)
    logs=list((ROOT/'net_results').glob('net-28069793_*.log'))
    assert len(logs)==24
    for p in logs:
        lines=p.read_text().splitlines();v=json.loads(lines[0]);end=json.loads(lines[-1])
        assert not v['missing'] and not v['changed'] and v['net_verified'] and v['bot1_native_loaded'] and end['complete']
    indices=np.random.default_rng(20261106).integers(0,64,(20000,64))
    contrast={}
    for payout in ('top3','winner','top4'):
        means=[]
        for rows in (net,ppo):
            subset=[r for r in rows if r['payout']==payout and 0 in r['seats']]
            means.append(np.array([np.mean([r['points'][r['seats'].index(0)] for r in subset if r['seed_index']==s]) for s in range(64)]))
        contrast[payout]=summarize(means[0]-means[1],indices)
    (ROOT/'net_vs_ppo.json').write_text(json.dumps({'net_minus_ppo':contrast,'design':'Same seed/field/rotation schedule in separate runs, not direct head-to-head. Hardware and timing-dependent sampling differ. Seed-block bootstrap; marginal intervals.'},indent=2))
    lines=['# Pure-net Claude comparison','','13,824 additional eight-player tournaments on Engaging, November 6, 2026. The eight original bots were frozen. The baseline used CFR-distilled `policy_v9` with `postflop_v6s` and `postflop3_v1`, no PPO fine-tuning, no supplied heuristic overrides. The original default heuristic fallback remains outside supported net situations.','','## Matched comparison against each bot','','50/30/20 payouts. Positive numbers below mean the pure-net Claude baseline earned more prize-pool share. Each comparison covers 3,584 tables containing both players; uncertainty resamples 64 whole seed blocks, including all seat rotations and relevant opponent fields. These are mixed eight-player tables, not heads-up games.','','| Opponent | Net advantage (percentage points) | 95% interval |','|---|---:|---|']
    for b in range(1,9):
        v=nsummary['payouts']['top3']['paired_vs_claude'][str(b)]
        lines.append(f"| bot_{b} | {-100*v['mean']:+.2f} | {-100*v['ci95'][1]:+.2f} to {-100*v['ci95'][0]:+.2f} |")
    lines+=['','## Pure net versus previous PPO experiment','','Average baseline share across balanced mixed fields; 4,096 tables per baseline per payout. The contrast uses matching seed blocks across separate experiments, not tables containing both baseline variants.','','| Payout | Pure net share | PPO share | Net minus PPO (pp), 95% interval |','|---|---:|---:|---|']
    for payout,label in [('top3','50/30/20'),('winner','Winner takes all'),('top4','40/30/20/10')]:
        n=nsummary['payouts'][payout]['mixed_fields']['0'];p=psummary['payouts'][payout]['mixed_fields']['0'];d=contrast[payout]
        lines.append(f"| {label} | {100*n['mean']:.2f}% | {100*p['mean']:.2f}% | {100*d['mean']:+.2f} [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}] |")
    lines+=['','## Audit and limits','','- All 24 shards completed; frozen source/model hashes verified on every shard. All matches have unique complete schedule IDs, correct chip/prize totals, and zero recorded exceptions, illegal actions or internal errors.','- Identical experimental engine, initial stacks, blinds, payout sensitivity and 1 ms check/fold fallback as the prior experiment. See `RESULTS.md` for full design.','- Source pinned to `a3d75f7c736da8c69e474ceed7e189682f9e6c7e`. `net_baseline_manifest.json` contains selected weights/configuration and hashes; `build_net_baseline.py` reconstructs it from that repository.','- The pure-net models use solver-generated targets. This is not a claim of exact full-game equilibrium or absence of heuristic runtime choices.','- Hardware nodes and timing-dependent action sampling vary; seed pairing controls deck schedules, not all environmental randomness. Marginal 95% intervals are not adjusted for multiple comparisons.','- No official starter kit or payout schedule was available. Results apply to these experimental fields and rules.','','## Timing','','| Bot | Decisions | Over 1 ms | Fallback rate |','|---|---:|---:|---:|']
    for b in range(9):
        t=nsummary['timing'][str(b)];lines.append(f"| {'Claude pure net' if b==0 else f'bot_{b}'} | {t['decisions']:,} | {t['timeouts']:,} | {100*t['timeout_fraction']:.4f}% |")
    (ROOT/'NET_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with gzip.open(ROOT/'net_results.jsonl.gz','wb') as f:
        for p in sorted((ROOT/'net_results').glob('production_*.jsonl')):f.write(p.read_bytes())
    (ROOT/'net_audit.json').write_text(json.dumps({'tables':13824,'shards':24,'logs_verified':True,'all_frozen_hashes_pass':True,'decisions':sum(v['decisions'] for v in nsummary['timing'].values()),'timeouts':sum(v['timeouts'] for v in nsummary['timing'].values()),'faults':0,'raw_sha256':hashlib.sha256((ROOT/'net_results.jsonl.gz').read_bytes()).hexdigest()},indent=2))
    print(json.dumps(contrast))

if __name__=='__main__':main()
