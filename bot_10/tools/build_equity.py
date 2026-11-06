"""Reproduce offline preflop assets; requires GCC for generation only.

python tools/build_equity.py --samples 50000 --seed 9173248901
Each estimate samples random boards and uniformly random legal opponent combos
in a specified range. Ranges rank canonical classes by first-pass uniform equity
and include classes until at least 35%, 15%, or 5% of 1326 combos are covered.
These are modeling assumptions, not official opponent distributions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=9173248901)
    parser.add_argument("--output", default="assets/preflop_equity.json")
    args = parser.parse_args()
    if args.samples < 100:
        parser.error("samples must be at least 100")
    exe = ROOT / "tools" / ("equity_native.exe" if sys.platform == "win32" else "equity_native")
    subprocess.run(["gcc", "-O3", "-std=c99", str(ROOT / "tools/equity_native.c"), "-o", str(exe)], check=True)
    # Check the native sampler's rank against the Python implementation.
    rng = random.Random(720518)
    hands = [rng.sample(range(52), 7) for _ in range(10000)]
    data = "".join(" ".join(map(str, h)) + "\n" for h in hands)
    out = subprocess.run([str(exe), "verify"], input=data, text=True, capture_output=True, check=True)
    assert [int(x) for x in out.stdout.split()] == [evaluate(h) for h in hands], "native evaluator disagreement"
    start = time.perf_counter()
    result = subprocess.run([str(exe), str(args.samples), str(args.seed)], text=True, capture_output=True, check=True)
    ranks = "23456789TJQKA"
    table = {}
    for line in result.stdout.splitlines():
        cid, profile, equity = line.split()
        a, b = divmod(int(cid), 13)
        hi, lo = max(a, b), min(a, b)
        key = ranks[hi] + ranks[lo] + ("" if a == b else "s" if a > b else "o")
        table.setdefault(key, [None] * 4)[int(profile)] = float(equity)
    assert len(table) == 169 and all(None not in row for row in table.values())
    metadata = {
        "description": "Original Monte Carlo estimates; heads-up split-pot equity, card removal applied.",
        "profiles": ["uniform", "top35", "top15", "top5"],
        "samples_per_class_per_profile": args.samples,
        "seed": args.seed,
        "worst_case_standard_error": 0.5 / args.samples ** 0.5,
        "range_construction": "Classes ranked by sampled uniform equity; include full classes through desired combo fraction.",
        "actual_range_sizes": result.stderr.strip().splitlines(),
        "native_python_crosscheck_hands": len(hands),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "source_sha256": hashlib.sha256((ROOT / "tools/equity_native.c").read_bytes()).hexdigest(),
    }
    path = ROOT / args.output
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps({"metadata": metadata, "equities": dict(sorted(table.items()))}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
