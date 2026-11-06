"""Independent Python audit of the original C++ Monte Carlo dataset.

Exact heads-up river targets are recomputed from every legal combination.
Selected flop/turn cases receive an independent, higher-sample Python simulation.
This checks the whole dataset path, not just hand-rank agreement.
"""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import random
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bot.evaluator import evaluate
from bot.features import hand_index, WINDOWS


def pressure_weight(hole, board, pressure):
    if not pressure:
        return 1.
    ranks = [c // 4 + 2 for c in board]
    br = [ranks.count(r) for r in range(15)]
    suit_counts = [sum(c % 4 == s for c in board) for s in range(4)]
    a, z = [c // 4 + 2 for c in hole]
    score = evaluate(list(hole) + board)
    cat, primary = score >> 20, (score >> 16) & 15
    if cat == 0:
        w = .16 if 14 in (a, z) else .08
    elif cat == 1:
        if a == z == primary or br[primary] == 1:
            w = .95 if primary >= max(ranks) else .60 if primary >= max(ranks) - 3 else .38
        else:
            w = .22 if max(a, z) == 14 else .12
    elif cat == 2:
        w = .95 if br[a] + br[z] + (2 if a == z else 0) else .18
    elif cat == 3:
        w = 1. if br[primary] < 3 else .25
    else:
        w = .32 if len(board) == 5 and evaluate(board) == score else 1.
    if len(board) < 5:
        for c in hole:
            suit_counts[c % 4] += 1
        if any(suit_counts[c % 4] == 4 for c in hole):
            w = max(w, .65)
        mask = 0
        for c in list(hole) + board:
            mask |= 1 << (c // 4)
        outs = 0
        for win in WINDOWS:
            if (mask & win).bit_count() == 4:
                outs |= win & ~mask
        count = outs.bit_count()
        if count >= 2:
            w = max(w, .55)
        elif count == 1:
            w = max(w, .30)
    return w if pressure == 1 else max(.015, w ** 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='training/review.csv')
    p.add_argument('--strength', default='assets/strength.txt')
    p.add_argument('--river-cases', type=int, default=400)
    p.add_argument('--mc-cases', type=int, default=18)
    p.add_argument('--mc-samples', type=int, default=6000)
    p.add_argument('--output', default='benchmarks/generator_audit.json')
    args = p.parse_args()
    strengths = list(map(float, open(args.strength)))
    combos = [(a, b) for a in range(52) for b in range(a + 1, 52)]
    combos.sort(key=lambda h: strengths[hand_index(h)], reverse=True)
    ranges = {}
    for f in (.1, .25, .5, 1.):
        cutoff = strengths[hand_index(combos[math.ceil(1326 * f) - 1])]
        ranges[f] = [h for h in combos if strengths[hand_index(h)] + 1e-9 >= cutoff]
    rows = [[float(x) for x in row] for row in csv.reader(open(args.dataset))]
    exact = []
    mc = []
    rng = random.Random(161112)
    for r in rows:
        hole = [int(c) for c in r[:2]]
        board = [int(c) for c in r[2:7] if c >= 0]
        n, fraction, pressure, target = int(r[7]), r[8], int(r[9]), r[10]
        assert len(set(hole + board)) == len(hole + board)
        assert 0 <= target <= 1
        river = len(board) == 5 and n == 1 and len(exact) < args.river_cases
        simulate = len(board) < 5 and len(mc) < args.mc_cases
        if not river and not simulate:
            continue
        known = set(hole + board)
        candidates = [h for h in ranges[fraction] if not known.intersection(h)]
        weights = [pressure_weight(h, board, pressure) for h in candidates]
        if river:
            hv = evaluate(hole + board)
            scores = [evaluate(list(h) + board) for h in candidates]
            expected = sum(w * (1 if hv > v else .5 if hv == v else 0) for w, v in zip(weights, scores)) / sum(weights)
            error = expected - target
            if abs(error) > 1e-7:
                raise AssertionError((r, expected))
            exact.append({'pressure': pressure, 'error': error})
        else:
            cdf = []
            total_weight = 0
            for w in weights:
                total_weight += w
                cdf.append(total_weight)
            value = 0.
            completed = 0
            while completed < args.mc_samples:
                used = set(known)
                villains = []
                failed = False
                for j in range(n):
                    for attempt in range(3000):
                        h = candidates[bisect.bisect_left(cdf, rng.random() * total_weight)]
                        if not used.intersection(h):
                            villains.append(h)
                            used.update(h)
                            break
                    else:
                        failed = True
                        break
                if failed:
                    continue
                runout = board + rng.sample([c for c in range(52) if c not in used], 5 - len(board))
                hv = evaluate(hole + runout)
                scores = [evaluate(list(h) + runout) for h in villains]
                if max(scores) <= hv:
                    value += 1 / (1 + scores.count(hv))
                completed += 1
            expected = value / args.mc_samples
            mc.append({'hole': hole, 'board': board, 'opponents': n, 'range': fraction,
                       'pressure': pressure, 'cpp_target': target, 'python_target': expected,
                       'difference': expected - target})
    errors = [v['difference'] for v in mc]
    report = {'dataset': args.dataset, 'row_count': len(rows),
              'exact_river_cases': len(exact), 'exact_river_max_error': max(abs(v['error']) for v in exact),
              'exact_by_pressure': {str(k): sum(v['pressure'] == k for v in exact) for k in (0, 1, 2)},
              'python_mc_samples_per_case': args.mc_samples, 'python_mc_seed': 161112,
              'mc_rmse_difference': math.sqrt(statistics.mean(e * e for e in errors)),
              'mc_max_difference': max(abs(e) for e in errors), 'mc_cases': mc}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'mc_cases'}, indent=2))


if __name__ == '__main__':
    main()
