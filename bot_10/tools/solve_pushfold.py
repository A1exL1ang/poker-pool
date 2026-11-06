"""Reproduce a chip-EV shove/call table for a documented first-caller model.

python tools/solve_pushfold.py --iterations 500

The original C++ helper uses one CPU. Runtime needs only the generated JSON.
This is an approximate small-game solution, not a full Hold'em equilibrium.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEPTHS = (2, 3, 4, 6, 8, 10, 12, 15)


def action_values(equity, conditional, shove, calls, depth, pot=1.5):
    """Independent stdlib reference for values/regrets; used by verification."""
    n, m = len(shove), len(calls)
    hero_cost = depth - (.5 if m == 1 else 0)
    caller_costs = [depth - (1 if j == m - 1 else .5 if j == m - 2 else 0) for j in range(m)]
    reach = [[0.] * n for _ in calls]
    opener = [0.] * n
    for h in range(n):
        remaining = 1.
        for j in range(m):
            reach[j][h] = remaining
            mass = sum(p * c for p, c in zip(conditional[h], calls[j]))
            wins = sum(p * c * q for p, c, q in zip(conditional[h], calls[j], equity[h]))
            opener[h] += remaining * (wins * (pot + hero_cost + caller_costs[j]) - mass * hero_cost)
            remaining *= 1 - mass
        opener[h] += remaining * pot
    caller = [[sum(conditional[c][h] * shove[h] * reach[j][h] *
                   (equity[c][h] * (pot + hero_cost + caller_costs[j]) - caller_costs[j])
                   for h in range(n)) for c in range(n)] for j in range(m)]
    return opener, caller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", default="assets/pushfold.json")
    args = parser.parse_args()
    if args.iterations < 100 or args.iterations > 10000:
        parser.error("iterations must be 100..10000")
    source = ROOT / "tools/pushfold_native.cpp"
    exe = ROOT / "tools" / ("pushfold_native.exe" if sys.platform == "win32" else "pushfold_native")
    subprocess.run(["g++", "-O3", "-std=c++17", str(source), "-o", str(exe)], check=True)
    raw = (ROOT / "assets/preflop_matchups.json").read_bytes()
    matchups = json.loads(raw)
    prior = [count / 1326 for count in matchups["combo_counts"]]
    conditional = [[count / 1225 for count in row] for row in matchups["legal_opponent_counts"]]
    data = "169\n" + " ".join(map(str, prior)) + "\n"
    data += "\n".join(" ".join(map(str, row)) for row in conditional) + "\n"
    data += "\n".join(" ".join(map(str, row)) for row in matchups["equity"]) + "\n"
    start = time.perf_counter()
    result = subprocess.run([str(exe), str(args.iterations)], input=data, text=True, capture_output=True, check=True)
    shove, call, diagnostics = {}, {}, []
    for line in result.stdout.splitlines():
        values = list(map(float, line.split()))
        depth, behind = map(int, values[:2])
        frequency, delta, identity_error = values[2:5]
        residuals = values[5:6 + behind]
        offset = 6 + behind
        s = values[offset:offset + 169]
        cs = [values[offset + 169 + j * 169:offset + 169 + (j + 1) * 169] for j in range(behind)]
        assert len(s) == 169 and all(len(c) == 169 for c in cs)
        assert all(0 <= p <= 1 for p in s + sum(cs, []))
        shove.setdefault(str(depth), {})[str(behind)] = [round(x, 7) for x in s]
        call.setdefault(str(depth), {})[str(behind)] = [[round(x, 7) for x in c] for c in cs]
        # Independent Python implementation checks selected cases, not just outputs.
        if depth in (2, 8, 15) and behind in (1, 3, 7):
            ov, cv = action_values(matchups["equity"], conditional, s, cs, depth)
            rr = [sum(p * (max(v, 0) - x * v) for p, v, x in zip(prior, ov, s))]
            rr += [sum(p * (max(v, 0) - x * v) for p, v, x in zip(prior, row, strat)) for row, strat in zip(cv, cs)]
            assert max(abs(a - b) for a, b in zip(rr, residuals)) < 1e-9
        assert identity_error < 1e-9, "chip accounting identity failed"
        diagnostics.append({"depth_bb": depth, "behind": behind, "shove_frequency": frequency,
                            "opener_max_strategy_change_last50": delta, "regret_bb_per_dealt_hand": residuals,
                            "max_player_regret_bb": max(residuals), "chip_accounting_error": identity_error})
    assert len(diagnostics) == 56
    metadata = {
        "description": "Approximate chip-EV fictitious-play strategies in a first-caller preflop abstraction.",
        "iterations": args.iterations,
        "depths_bb": DEPTHS,
        "seats_behind": list(range(1, 8)),
        "algorithm": "Simultaneous best responses to average strategies, running-average step 1/(iteration+2). Initial shove .25 and call .15 for all classes.",
        "abstraction": "Equal starting stacks in BB; opener fold-or-shove; at most first caller; later seats forced fold. Opponent classes sampled independently conditional on opener class and exact opener-card blockers. No card-removal interaction between opponents. Later caller posteriors include earlier fold reach.",
        "chip_accounting": "Pot 1.5BB; hero prior contribution .5 only with one seat behind; last caller BB contributes1, penultimate SB .5, others0. Fold is zero additional EV. Uncalled shove gains1.5. Called opener EV=q*(1.5+hero_remaining+caller_remaining)-hero_remaining.",
        "call_arrays": "call[depth][behind][caller_index][class_index]; caller index is order behind opener, last is BB.",
        "limitations": "Not an equilibrium claim for full Hold'em or proof of convergence; no limps, minraises, overcalls, unequal stacks, antes, ICM, conditional folds before opener, or postflop play. Monte Carlo payoff matrix has sampling error.",
        "regret_definition": "Exact unilateral best-response gain for each player against the resulting independently mixed strategy profile, BB per dealt hand under this abstraction. Player0 opener, followed by callers in order.",
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "matchups_sha256": hashlib.sha256(raw).hexdigest(),
        "max_player_regret_bb": max(d["max_player_regret_bb"] for d in diagnostics),
        "mean_max_player_regret_bb": sum(d["max_player_regret_bb"] for d in diagnostics) / len(diagnostics),
        "trace": result.stderr.strip().splitlines(),
    }
    output = {"metadata": metadata, "keys": matchups["keys"], "shove": shove, "call": call, "diagnostics": diagnostics}
    (ROOT / args.output).write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k != "trace"}, indent=2))
    print("Largest residuals:", sorted(((d["max_player_regret_bb"], d["depth_bb"], d["behind"]) for d in diagnostics), reverse=True)[:8])


if __name__ == "__main__":
    main()
