"""Policy model selection on explicit local tuning seeds only.

Every payout vector here is an experimental stress case, never official.
Fork workers after loading the immutable model on Linux to share its memory.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import statistics
import time

from bot.agent import Agent, Config
from benchmarks.tournament import evaluate_candidate, summary


CANDIDATES = {
    'default': {},
    'tight': dict(open_scale=.80, call_margin=.045, range_pressure=1.15, bluff_scale=.5),
    'loose': dict(open_scale=1.20, call_margin=0, range_pressure=.7, bluff_scale=.8),
    'no_pressure': dict(range_pressure=0),
    'half_pressure': dict(range_pressure=.5),
    'soft_pressure': dict(range_pressure=.8),
    'hard_pressure': dict(range_pressure=1.2),
    'no_bluffs': dict(bluff_scale=0),
    'no_adaptation': dict(adaptation=False),
    'no_icm': dict(icm_scale=0),
    'wide_open': dict(open_scale=1.25),
    'narrow_open': dict(open_scale=.8),
}
PAYOUTS = {
    'winner': [1., 0., 0., 0., 0., 0., 0., 0.],
    'top3': [.5, .3, .2, 0., 0., 0., 0., 0.],
}
POPULATIONS = ('mixed', 'tight', 'loose', 'aggressive')


def task(args):
    name, changes, population, payout, tables, seed = args
    cfg = Config(**changes)
    # Preserve original tuning experiment for reproduction. It rotated seats
    # in lockstep with the initial button; final held-out evaluation corrects
    # this correlation without changing the selected policy.
    rows = evaluate_candidate(lambda s: Agent(s, cfg), tables, seed, population,
                              payouts=PAYOUTS[payout], seat_mode='legacy')
    return dict(candidate=name, population=population, payout=payout, config=asdict(cfg),
                score=summary([r['points'] * 8 for r in rows]), rows=rows,
                errors=sum(r['errors'] + r['invalid_actions'] for r in rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tables', type=int, default=96)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--stage', choices=('initial', 'refinement'), default='initial')
    p.add_argument('--output', default='artifacts/policy_tuning.json')
    args = p.parse_args()
    Agent()  # preload outside jobs; immutable equity arrays shared under fork
    candidates = CANDIDATES if args.stage == 'initial' else {
        **{f'open{width}_pressure{pressure}': dict(open_scale=width, range_pressure=pressure)
           for width in (.8, 1.0, 1.2) for pressure in (1.0, 1.2, 1.4)},
        'pressure1.2_no_adaptation': dict(range_pressure=1.2, adaptation=False),
    }
    seed_base = 210001 if args.stage == 'initial' else 310001
    tasks = []
    for name, changes in candidates.items():
        for i, population in enumerate(POPULATIONS):
            for j, payout in enumerate(PAYOUTS):
                # Pair candidate policies on exactly the same per-hand decks.
                tasks.append((name, changes, population, payout, args.tables, seed_base + i * 10000 + j * 1000))
    started = time.time()
    ctx = multiprocessing.get_context('fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn')
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for row in pool.map(task, tasks):
            results.append(row)
            print(row['candidate'], row['population'], row['payout'], round(row['score']['mean'], 4), flush=True)
            Path(args.output + '.checkpoint').write_text(json.dumps(results))
    scores = []
    for name in candidates:
        selected = [r for r in results if r['candidate'] == name]
        means = [r['score']['mean'] for r in selected]
        # Predeclared robust score: equal-weight scenario mean with 20% weight
        # on the weakest population (averaged over both payout stress cases).
        popmeans = [statistics.fmean(r['score']['mean'] for r in selected if r['population'] == pop) for pop in POPULATIONS]
        objective = .8 * statistics.fmean(means) + .2 * min(popmeans)
        scores.append(dict(candidate=name, objective=objective, mean=statistics.fmean(means),
                           worst_population=min(popmeans), config=selected[0]['config']))
    scores.sort(key=lambda r: r['objective'], reverse=True)
    report = dict(split='tuning_only', stage=args.stage, seed_base=seed_base, tables_per_cell=args.tables,
                  objective='0.8 * scenario mean + 0.2 * weakest population mean; scores normalized by equal-share points',
                  assumptions='Independent local baselines, doubling blind schedule, illustrative payouts only. Legacy seating fixes initial relative position; corrected in final held-out evaluation.',
                  seconds=time.time() - started, ranking=scores, results=results)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == '__main__':
    main()
