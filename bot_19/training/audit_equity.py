"""Audit range direction, shared boards, and HU-to-multiway approximation.

This is synthetic calibration evidence, not held-out tournament evaluation.
"""
import argparse
import json
import math
import random
import subprocess
from pathlib import Path


def cards(text):
    return ["23456789TJQKA".index(c[0]) * 4 + "cdhs".index(c[1]) for c in text.split()]


def request(h, b, w, a, n, samples, seed):
    return f"{len(b)} " + " ".join(map(str, h+b)) + f" {w} {a} {n} {samples} {seed}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    parser.add_argument("base")
    parser.add_argument("--cases", type=int, default=450)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--output", default="measurements/equity_range_audit.json")
    parser.add_argument("--data-output", default="measurements/equity_multiway_tuning_cases.json")
    parser.add_argument("--seed", type=int, default=1907719)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    cases = []
    inputs = []
    ns = [1, 2, 3, 5, 7]
    for i in range(args.cases):
        nb = 3 + i % 3
        hand = rng.sample(range(52), nb+2)
        w = rng.choice((.10, .35, 1.0))
        a = rng.choice((0, .5, 1))
        cases.append((nb, hand[:2], hand[2:], w, a))
        for n in ns:
            inputs.append(request(hand[:2], hand[2:], w, a, n, args.samples, args.seed*31+i*7+n))
    special = [
        ("royal_board", "2c 3d", "Ah Kh Qh Jh Th", .5),
        ("unflushable_broadway_board", "2c 3d", "Ah Kc Qd Js Th", .5),
        ("quads_ace_kicker_board", "2c 3d", "9c 9d 9h 9s Ah", .5),
        ("private_royal", "Ah Kh", "Qh Jh Th 2c 3d", 1.0),
        ("private_nut_quads", "Ac Ad", "Ah As Kh 2c 3d", 1.0),
    ]
    for _, h, b, _ in special:
        inputs.append(request(cards(h), cards(b), 1, 0, 1, args.samples, 190077))
    result = subprocess.run([args.executable, "query", args.base], input="".join(inputs),
                            capture_output=True, text=True, check=True)
    values = list(map(float, result.stdout.split()))
    if len(values) != len(inputs):
        raise RuntimeError("native query output mismatch")
    checks = []
    for (name, h, b, expected), actual in zip(special, values[len(cases)*len(ns):]):
        if abs(expected-actual) > 1e-9:
            raise AssertionError((name, expected, actual))
        checks.append({"name": name, "equity": actual, "expected": expected})
    scores = {}
    for p in (.70, .75, .80, .85, .90, .95, 1.0, 1.05):
        errors = {"all": [], "flop": [], "turn": [], "river": [], "hu_lt_050": [], "hu_ge_050": []}
        for i, (nb, h, b, w, a) in enumerate(cases):
            eq = values[i*len(ns): (i+1)*len(ns)]
            for j, n in enumerate(ns[1:], 1):
                error = eq[0] ** (n**p) - eq[j]
                errors["all"].append(error)
                errors[{3:"flop",4:"turn",5:"river"}[nb]].append(error)
                errors["hu_lt_050" if eq[0] < .5 else "hu_ge_050"].append(error)
        scores[str(p)] = {key: {"rmse": math.sqrt(sum(e*e for e in es)/len(es)),
                               "bias": sum(es)/len(es), "mae": sum(abs(e) for e in es)/len(es)}
                         for key, es in errors.items()}
    report = {"kind": "synthetic calibration; not final held-out evaluation", "seed": args.seed,
              "cases": args.cases, "samples_per_multiway_case": args.samples, "opponents": ns,
              "range_widths": [.10,.35,1.0], "aggressions": [0,.5,1], "special_cases": checks,
              "power_formula": "exact_or_high_sample_HU_equity ** (opponents ** power)", "scores": scores,
              "limitation": "identical ranges, sequential conditional card removal, no side pots or action selection"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
    data = {"seed": args.seed, "samples": args.samples, "opponents": ns,
            "cases": [{"hole": h, "board": b, "width": w, "aggression": a,
                       "equity": values[i*len(ns):(i+1)*len(ns)]}
                      for i, (nb, h, b, w, a) in enumerate(cases)]}
    Path(args.data_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.data_output).write_text(json.dumps(data, separators=(",", ":"))+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
