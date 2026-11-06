"""Independent, exact river equity calibration against all legal hole pairs.

The reference here uses the separately tested Python evaluator, and reconstructs
range weights from scratch. No training labels are reused.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot.cards import evaluate, features, preflop_index
from bot.model import PREFLOP, predict


def exact(hole, board):
    deck = [c for c in range(52) if c not in hole and c not in board]
    hero = evaluate((*hole, *board))
    trials = []
    current = 0.
    for a, b in itertools.combinations(deck, 2):
        rank = evaluate((a, b, *board))
        result = 1. if hero > rank else .5 if hero == rank else 0.
        current += result
        trials.append((rank, result, PREFLOP['percentile'][preflop_index((a, b))]))
    trials.sort()
    sums = [0.] * 8
    wins = [0.] * 8
    start = 0
    while start < len(trials):
        end = start + 1
        while end < len(trials) and trials[start][0] == trials[end][0]:
            end += 1
        q = (start + end) / 2 / len(trials)
        for _, result, pf in trials[start:end]:
            weights = [0., 1., float(pf < .5), float(pf < .2), float(pf < .05),
                       .08 + .92 * min(1., max(0., (q-.40)/.25)),
                       .04 + .96 * min(1., max(0., (q-.70)/.20)),
                       .025 + .975 * min(1., max(0., (q-.85)/.12))]
            for k in range(1, 8):
                sums[k] += weights[k]
                wins[k] += weights[k] * result
        start = end
    return [current / len(trials)] + [wins[k]/sums[k] for k in range(1, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', type=int, default=600)
    ap.add_argument('--seed', type=int, default=441006)
    ap.add_argument('--output', default='evidence/exact_river_calibration.json')
    args = ap.parse_args()
    rng = random.Random(args.seed)
    squared, absolute, bias = [0.]*8, [0.]*8, [0.]*8
    worst = []
    start = time.perf_counter()
    for i in range(args.states):
        cs = rng.sample(range(52), 7)
        p = predict(features(cs[:2], cs[2:]))
        y = exact(cs[:2], cs[2:])
        for k in range(8):
            error = p[k]-y[k]
            squared[k] += error*error
            absolute[k] += abs(error)
            bias[k] += error
        worst.append((max(abs(a-b) for a,b in zip(p,y)), cs, p, y))
    worst.sort(reverse=True)
    report = {'states': args.states, 'seed': args.seed, 'hole_pairs_per_state': 990,
              'rmse': [math.sqrt(x/args.states) for x in squared],
              'mae': [x/args.states for x in absolute], 'bias': [x/args.states for x in bias],
              'seconds': time.perf_counter()-start, 'worst_ten': worst[:10]}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='worst_ten'}, indent=2))


if __name__ == '__main__':
    main()
