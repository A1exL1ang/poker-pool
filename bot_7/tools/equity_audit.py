"""River calibration against exact enumeration of the agent's declared ranges."""
import json
from pathlib import Path
import random
import statistics
from bot7.equity import COMBOS, COMBO_MASK, equity
from bot7.evaluator import evaluate


def exact(hole, board, fraction, aggression):
    dead = sum(1 << c for c in hole + board)
    hero = evaluate(hole + board)
    top = max(c // 4 + 2 for c in board)
    paired = len({c // 4 for c in board}) < len(board)
    won = mass = 0.0
    for i, (a, b) in enumerate(COMBOS[:max(24, min(1326, int(fraction * 1326)))]):
        if COMBO_MASK[i] & dead:
            continue
        score = evaluate((a, b) + board)
        cat, rank = score >> 20, (score >> 16) & 15
        strength = (.98 if cat >= 3 else .9 if cat == 2 and not paired else .65 if cat == 2
                    else .80 if cat == 1 and rank >= top else .4 if cat == 1 and not paired
                    else .24 if cat == 1 else .14)
        weight = 1.0 if aggression == 0 else .18 + .82 * strength if aggression == 1 else .06 + .94 * strength * strength
        mass += weight
        won += weight * (1 if hero > score else .5 if hero == score else 0)
    return won / mass if mass else None


def main():
    rng = random.Random(7017)
    results = []
    for case in range(120):
        cards = rng.sample(range(52), 7)
        hole, board = tuple(cards[:2]), tuple(cards[2:])
        fraction = [1, .4, .1, .025][case % 4]
        aggression = case % 3
        target = exact(hole, board, fraction, aggression)
        if target is None:
            continue
        estimates = [equity(hole, board, [(fraction, aggression)], random.Random(case * 1000 + seed)) for seed in range(80)]
        results.append({"case": case, "fraction": fraction, "aggression": aggression, "exact": target,
                        "mean_estimate": statistics.mean(estimates), "single_decision_rmse": (statistics.mean((x - target) ** 2 for x in estimates)) ** .5})
    bias = statistics.mean(r["mean_estimate"] - r["exact"] for r in results)
    report = {"cases": len(results), "estimates_per_case": 80, "comparisons": 48,
              "mean_signed_error": bias,
              "mean_absolute_error_of_80_trial_means": statistics.mean(abs(r["mean_estimate"] - r["exact"]) for r in results),
              "pooled_single_decision_rmse": statistics.mean(r["single_decision_rmse"] ** 2 for r in results) ** .5,
              "note": "Exact only against the declared heuristic river ranges, not real opponent ranges. Sampling error is material near decision thresholds.",
              "results": results}
    assert abs(bias) < .025
    Path("benchmarks/equity_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
