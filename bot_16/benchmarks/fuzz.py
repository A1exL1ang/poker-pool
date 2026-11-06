"""Bounded randomized mechanics stress, NOT a strategy benchmark."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator import play_hand
from baselines import Baseline
from benchmarks.tournament import fingerprints


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hands', type=int, default=10000)
    p.add_argument('--seed', type=int, default=160701)
    p.add_argument('--output', default='benchmarks/mechanics_fuzz.json')
    p.add_argument('--stress', action='store_true', help='include separate adversarial DEVELOPMENT opponents')
    args = p.parse_args()
    rng = random.Random(args.seed)
    styles = Baseline.STYLES
    if args.stress:
        from baselines_stress import StressBaseline
        styles += ('minraise', 'calling', 'polar')
    def make_agent(style, seed):
        return Baseline(style, seed) if style in Baseline.STYLES else StressBaseline(style, seed)
    violations = decisions = 0
    start = time.time()
    hashes = fingerprints()
    for h in range(args.hands):
        n = rng.randrange(2, 9)
        stacks = [rng.randrange(1, 5001) for _ in range(n)]
        blind = rng.choice([2, 10, 20, 100, 1000])
        agents = [make_agent(rng.choice(styles), h * 8 + i) for i in range(n)]
        result = play_hand(agents, stacks, button=h % n, big_blind=blind,
                           small_blind=blind // 2, seed=h + args.seed)
        assert sum(result.stacks) == sum(stacks)
        violations += len(result.violations)
        decisions += sum(result.decisions_by_seat)
    assert not violations, violations
    report = dict(seed=args.seed, hands=args.hands, decisions=decisions, violations=violations,
                  chip_conservation_failures=0, elapsed_seconds=time.time() - start,
                  styles=styles, file_sha256=hashes,
                  files_changed_during_run=fingerprints() != hashes,
                  purpose='randomized mechanics stress, not strategic validation')
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
