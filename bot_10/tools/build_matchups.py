"""Build original blocker-aware 169x169 heads-up preflop matchup assets.

python tools/build_matchups.py --samples 10000 --seed 550109173
GCC is needed only for asset generation. Game runtime loads JSON only.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import preflop_key


def key_for_id(cid):
    a, b = divmod(cid, 13)
    return preflop_key(a * 4, b * 4 + (a <= b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=550109173)
    parser.add_argument("--output", default="assets/preflop_matchups.json")
    args = parser.parse_args()
    if args.samples < 100:
        parser.error("at least 100 samples required")
    exe = ROOT / "tools" / ("matchups_native.exe" if sys.platform == "win32" else "matchups_native")
    subprocess.run(["gcc", "-O3", "-std=c99", str(ROOT / "tools/matchups_native.c"), "-o", str(exe)], check=True)
    start = time.perf_counter()
    result = subprocess.run([str(exe), str(args.samples), str(args.seed)], text=True, capture_output=True, check=True)
    keys = sorted(key_for_id(cid) for cid in range(169))
    index = {k: i for i, k in enumerate(keys)}
    matrix = [[None] * 169 for _ in keys]
    legal_counts = [[0] * 169 for _ in keys]
    for line in result.stdout.splitlines():
        ci, cj, value, a, b = line.split()
        i, j = index[key_for_id(int(ci))], index[key_for_id(int(cj))]
        matrix[i][j] = float(value)
        matrix[j][i] = round(1 - float(value), 9)
        legal_counts[i][j], legal_counts[j][i] = int(a), int(b)
    combo_counts = [6 if len(k) == 2 else 4 if k.endswith("s") else 12 for k in keys]
    # Independently recompute concrete opponent counts in Python. Suit permutations
    # act transitively within each hero class, so one representative is sufficient.
    for cid in range(169):
        a, b = divmod(cid, 13)
        hole = (a * 4, b * 4 + (a <= b))
        expected = [0] * 169
        for pair in itertools.combinations((c for c in range(52) if c not in hole), 2):
            expected[index[preflop_key(pair)]] += 1
        i = index[key_for_id(cid)]
        assert expected == legal_counts[i]
        assert sum(legal_counts[i]) == 1225
        for j in range(169):
            assert combo_counts[i] * legal_counts[i][j] == combo_counts[j] * legal_counts[j][i]
            assert matrix[i][j] is not None and 0 <= matrix[i][j] <= 1
            assert abs(matrix[i][j] + matrix[j][i] - 1) < 1e-8
        assert matrix[i][i] == .5
    metadata = {
        "description": "Original heads-up preflop split-pot equity matrix. Row hero, column opponent.",
        "samples_per_unordered_distinct_class_pair": args.samples,
        "sampled_pair_count": 169 * 168 // 2,
        "sampled_showdowns": 169 * 168 // 2 * args.samples,
        "seed": args.seed,
        "worst_case_pair_standard_error": .5 / args.samples ** .5,
        "sampling": "Fix suit-equivalent hero representative; sample uniformly all legal concrete opponent combos of the requested class, then uniformly remaining board cards.",
        "symmetry": "Reverse entries are one minus forward; equal-class equity is exactly 0.5 by exchangeability.",
        "legal_opponent_counts": "For a fixed hero combo; rows each sum to C(50,2)=1225. Accounts exactly for hero card removal.",
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "source_sha256": {name: hashlib.sha256((ROOT / "tools" / name).read_bytes()).hexdigest() for name in ("matchups_native.c", "equity_native.c")},
    }
    old = json.loads((ROOT / "assets/preflop_equity.json").read_text())
    uniform = [sum(q * n for q, n in zip(matrix[i], legal_counts[i])) / 1225 for i in range(169)]
    differences = [uniform[i] - old["equities"][k][0] for i, k in enumerate(keys)]
    metadata["independent_uniform_validation"] = {
        "reference_samples_per_class": old["metadata"]["samples_per_class_per_profile"],
        "rmse": (sum(x * x for x in differences) / len(differences)) ** .5,
        "max_absolute_difference": max(map(abs, differences)),
        "comparison": "Blocker-weighted aggregate against independently sampled uniform-opponent estimates; no tuning to reference.",
    }
    path = ROOT / args.output
    path.write_text(json.dumps({"metadata": metadata, "keys": keys, "combo_counts": combo_counts,
                                "legal_opponent_counts": legal_counts, "equity": matrix}, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
