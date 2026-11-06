"""Cross-language card/feature parity and complete five-card category census."""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poker_eval import evaluate, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    parser.add_argument("--cases", type=int, default=20000)
    parser.add_argument("--output")
    args = parser.parse_args()
    rng = random.Random(190903)
    hands = [rng.sample(range(52), rng.choice((5, 6, 7))) for _ in range(args.cases)]
    inp = "".join(str(len(h)) + " " + " ".join(map(str, h)) + "\n" for h in hands)
    result = subprocess.run([args.executable, "check"], input=inp, text=True, capture_output=True, check=True)
    ranks = list(map(int, result.stdout.split()))
    assert len(ranks) == len(hands)
    assert ranks == [evaluate(h) for h in hands], "native rank mismatch"
    cases = [(rng.sample(range(52), rng.choice((5, 6, 7))), rng.uniform(.05, 1), rng.random()) for _ in range(args.cases)]
    inp = "".join(f"{len(h)-2} " + " ".join(map(str, h)) + f" {w:.12f} {a:.12f}\n" for h, w, a in cases)
    result = subprocess.run([args.executable, "features"], input=inp, text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    assert len(lines) == len(cases)
    for line, (h, w, a) in zip(lines, cases):
        actual = list(map(float, line.split()))
        expected = features(h[:2], h[2:], w, a)
        assert len(actual) == len(expected)
        assert max(abs(x-y) for x, y in zip(actual, expected)) < 1e-6, (h, actual, expected)
    result = subprocess.run([args.executable, "census"], text=True, capture_output=True, check=True)
    census = list(map(int, result.stdout.split()))
    assert census == [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40], census
    report = {"rank_parity_cases": len(hands), "feature_parity_cases": len(cases),
              "five_card_census": census, "five_card_hands": sum(census), "passed": True}
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
