"""Offline original HU shove/fold game trainer (not a full NLHE solution).

Example bounded compute-node command:
  python tools/train_pushfold.py --samples 20000 --threads 8 --iterations 20000

Samples are per class-pair cell, rounded up to equal samples per legal opposing
suit combination. The asset records actual counts, matrix Monte Carlo error,
and independently computable best-response gaps in the learned abstraction.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build(openmp=True):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise SystemExit("C++17 g++ or clang++ is required for offline training")
    windows = sys.platform == "win32"
    output = ROOT / "native" / ("b13_pushfold.exe" if windows else "b13_pushfold")
    base = [compiler, "-std=c++17", "-O3", "-DNDEBUG", "-Wall", "-Wextra", str(ROOT / "native" / "pushfold.cpp"), "-o", str(output)]
    if windows:
        base += ["-static-libgcc", "-static-libstdc++"]
    command = base + (["-fopenmp"] if openmp else [])
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode and openmp:
        print("OpenMP compilation unavailable; retrying the bounded serial trainer.", file=sys.stderr)
        subprocess.run(base, cwd=ROOT, check=True)
    elif result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return output


def validate_asset(data):
    n = len(data["classkeys"])
    if n != 169 or len(set(data["classkeys"])) != 169 or sum(data["class_combo_counts"]) != 1326:
        raise ValueError("invalid class definitions")
    counts, eq = data["conditional_combo_counts"], data["equity_matrix"]
    for i in range(n):
        if len(counts[i]) != n or sum(counts[i]) != 1225 or len(eq[i]) != n:
            raise ValueError("invalid blocker-conditioned probability mass")
        for j in range(n):
            if counts[i][j]*data["class_combo_counts"][i] != counts[j][i]*data["class_combo_counts"][j]:
                raise ValueError("blocker counts violate detailed balance")
            if not math.isfinite(eq[i][j]) or not 0 <= eq[i][j] <= 1 or abs(eq[i][j]+eq[j][i]-1)>1e-8:
                raise ValueError("invalid symmetric equity matrix")
    for scenario in data["depths"]:
        if any(len(scenario[k]) != n or any(not 0 <= v <= 1 for v in scenario[k]) for k in ("shove_probs", "call_probs")):
            raise ValueError("invalid strategy probability")
        if scenario["nash_conv_bb"] < 0 or not math.isfinite(scenario["nash_conv_bb"]):
            raise ValueError("invalid exploitability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=131313)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--target-gap", type=float, default=.002, help="stop after exact learned-game NashConv is at most this many BB")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "pushfold.json")
    parser.add_argument("--no-openmp", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error("threads must be 1..8")
    executable = build(openmp=not args.no_openmp)
    if args.self_test:
        subprocess.run([str(executable), "--self-test"], check=True, cwd=ROOT)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        subprocess.run([str(executable), str(args.samples), str(args.seed), str(args.iterations), str(args.threads), str(args.target_gap)], stdout=stream, cwd=ROOT, check=True)
    data = json.loads(temporary.read_text(encoding="utf-8"))
    validate_asset(data)
    temporary.replace(args.output)
    print("Saved", args.output, "maximum learned-game exploitability BB:", max(s["exploitability_bb"] for s in data["depths"]))


if __name__ == "__main__":
    main()
