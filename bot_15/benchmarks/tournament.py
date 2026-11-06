"""Reproducible local evaluation. No claimed official scoring compatibility.

python -m benchmarks.tournament --tables 100 --split heldout --output artifacts/eval.json
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from benchmarks.engine import play_table
from benchmarks.opponents import BaselineAgent


POPULATIONS = {
    'mixed': ['balanced', 'tight', 'calling', 'aggressive', 'shove', 'balanced', 'tight'],
    'tight': ['tight'] * 7,
    'loose': ['calling', 'random', 'aggressive', 'calling', 'random', 'aggressive', 'calling'],
    'aggressive': ['aggressive', 'shove', 'aggressive', 'balanced', 'shove', 'aggressive', 'balanced'],
}


def summary(values):
    n = len(values)
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(n) if n > 1 else 0.0
    return dict(n=n, mean=mean, standard_error=se,
                ci95_normal=[mean - 1.96 * se, mean + 1.96 * se])


def candidate_seat(table_index, legacy=False):
    """Cross every initial seat/button pair, unless reproducing legacy tuning."""
    return table_index % 8 if legacy else (table_index // 8) % 8


def evaluate_candidate(factory, tables=100, seed=900001, population='mixed',
                       max_hands=200, payouts=None, opponent_factory=None,
                       decision_limit_ns=None, observer=None, seat_mode='crossed'):
    rows = []
    styles = POPULATIONS[population]
    for k in range(tables):
        table_seed = seed + k
        seat = candidate_seat(k, seat_mode == 'legacy')
        agents = [BaselineAgent(style, seed=table_seed * 13 + i)
                  for i, style in enumerate(styles)]
        if opponent_factory:
            agents = [opponent_factory(i, table_seed) for i in range(7)]
        agents.insert(seat, factory(table_seed))
        result = play_table(agents, table_seed, max_hands=max_hands, payouts=payouts,
                            decision_limit_ns=decision_limit_ns, observer=observer)
        rows.append(dict(seed=table_seed, seat=seat, chips=result['stacks'][seat],
                         points=None if result['points'] is None else result['points'][seat],
                         survived=int(result['stacks'][seat] > 0),
                         outright_win=int(result['stacks'][seat] == 16000),
                         **{key: result[key] for key in ('hands', 'decisions', 'errors', 'invalid_actions', 'overruns')}))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=int, default=100)
    parser.add_argument('--split', choices=['tuning', 'heldout'], default='heldout')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--population', choices=list(POPULATIONS), default='mixed')
    parser.add_argument('--candidate', default='bot', choices=['bot', 'balanced', 'tight', 'calling', 'aggressive', 'shove', 'random', 'fold'])
    parser.add_argument('--paired', choices=['balanced', 'tight', 'calling', 'aggressive', 'shove'])
    parser.add_argument('--payouts', help='Comma-separated LOCAL experimental finishing points, 8 entries')
    parser.add_argument('--max-hands', type=int, default=200)
    parser.add_argument('--enforce-1ms', action='store_true')
    parser.add_argument('--output', default='artifacts/tournament.json')
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else (10001 if args.split == 'tuning' else 900001)
    payouts = [float(x) for x in args.payouts.split(',')] if args.payouts else None
    if args.candidate == 'bot':
        from bot.agent import Agent
        factory = lambda s: Agent(seed=s)
    else:
        factory = lambda s: BaselineAgent(args.candidate, s)
    start = time.time()
    rows = evaluate_candidate(factory, args.tables, seed, args.population, args.max_hands,
                              payouts, decision_limit_ns=1000000 if args.enforce_1ms else None)
    result = dict(config=vars(args), assumptions={'blind_schedule': 'LOCAL doubling every 25 hands, 10/20 through 1280/2560',
                  'payouts': payouts, 'official_interface': False, 'timing': 'wall clock; optionally enforced locally'},
                  seconds=time.time() - start, rows=rows,
                  metrics={key: summary([row[key] for row in rows])
                           for key in ('chips', 'survived', 'outright_win')},
                  totals={key: sum(row[key] for row in rows)
                          for key in ('decisions', 'errors', 'invalid_actions', 'overruns')})
    if payouts is not None:
        result['metrics']['points'] = summary([row['points'] for row in rows])
    if args.paired:
        reference = evaluate_candidate(lambda s: BaselineAgent(args.paired, s), args.tables, seed,
                                       args.population, args.max_hands, payouts)
        result['paired_reference_rows'] = reference
        result['paired_differences'] = {key: summary([a[key] - b[key] for a, b in zip(rows, reference)])
                                         for key in ('chips', 'survived', 'outright_win') + (('points',) if payouts else ())}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'paired_reference_rows')}, indent=2))


if __name__ == '__main__':
    main()
