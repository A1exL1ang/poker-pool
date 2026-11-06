"""Generate static, exact single-player ICM kernels for up to eight players.

Two mathematically equivalent implementations are emitted for validation and
measurement. The default uses exponential-race subset moments, reducing the
work from O(m*2**m) to O(2**m) with no runtime loops over subset transitions.

Let h be hero's chips, x_i the opponents' chips, and J the number of opponents
finishing AFTER hero in the Plackett--Luce finish-order model. Independent
exponential clocks with rates equal to chips produce exactly that model.
For any opponent subset S, Pr(hero beats all S)=h/(h+sum_{i in S}x_i). Therefore
E[binom(J,s)] is the sum of these ratios over all subsets of size s. Newton's
finite-difference identity expands the payoff p[m-J] in these binomial moments.
After subtracting last-place base p[m], the result is

  h * sum_{k=0..r} c_k * sum_{|A|=k} 1/(h + sum_{i not in A}x_i)
  c_k = sum_{i=k..r} (-1)**(i-k) * binom(m-k,i-k) * (p[i]-p[m])

where r is the last payout index differing from base. Winner-only uses one
term; top-three uses 29 terms with seven opponents. math.fsum limits rounding
from the alternating coefficients. This is exact algebraically, not an ICM
approximation; floating-point roundoff remains and is measured in tests.

The alternative positive-term DP kernels use q[A]=Pr(A finish first)/remaining
chips. A transition is q[A] = sum(q[A-i]*x_i)/remaining(A), removing repeated
divisions present in the original loop. They provide a stable cross-check and
an optional fallback for unusual numerical scales.

Run: python -m training.compile_icm
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
import platform
import random
import statistics
from time import perf_counter_ns


def _header(m, last, kind):
    lines = [f"def _{kind}_{m}_{last}(h, x, d):"]
    for j in range(m):
        lines.append(f"    x{j} = x[{j}]")
    for j in range(last + 1):
        lines.append(f"    d{j} = d[{j}]")
    return lines


def _remaining(m, mask):
    return " + ".join(["h"] + [f"x{j}" for j in range(m) if not mask & (1 << j)])


def _dp(m, last):
    lines = _header(m, last, "dp")
    lines.append(f"    q0 = 1.0 / ({_remaining(m, 0)})")
    by_rank = [["q0"]] + [[] for _ in range(last)]
    for mask in range(1, 1 << m):
        rank = mask.bit_count()
        if rank > last:
            continue
        incoming = " + ".join(f"q{mask ^ (1 << j)} * x{j}" for j in range(m) if mask & (1 << j))
        lines.append(f"    q{mask} = ({incoming}) / ({_remaining(m, mask)})")
        by_rank[rank].append(f"q{mask}")
    terms = [f"d{rank} * ({' + '.join(values)})" for rank, values in enumerate(by_rank)]
    lines.append(f"    return h * ({' + '.join(terms)})")
    return "\n".join(lines)


def _race(m, last):
    lines = _header(m, last, "race")
    for k in range(last + 1):
        parts = []
        for i in range(k, last + 1):
            coefficient = comb(m - k, i - k)
            term = f"d{i}" if coefficient == 1 else f"{coefficient} * d{i}"
            if not parts:
                parts.append(term)
            else:
                parts.append((" - " if (i - k) % 2 else " + ") + term)
        lines.append(f"    c{k} = {''.join(parts)}")
    terms = []
    for k in range(last + 1):
        group = [f"1.0 / ({_remaining(m, mask)})" for mask in range(1 << m) if mask.bit_count() == k]
        total = group[0] if len(group) == 1 else "_fsum((" + ", ".join(group) + "))"
        terms.append(f"c{k} * ({total})")
    lines.append(f"    return h * _fsum(({', '.join(terms)}{',' if len(terms) == 1 else ''}))")
    return "\n".join(lines)


def render():
    pieces = ['''"""Generated exact ICM kernels. Regenerate with python -m training.compile_icm.

No dynamic compilation, filesystem access or external dependencies at decision
time. Default kernels use the exponential-race finite-difference identity;
positive-term unrolled DP is retained for independent validation and difficult
numerical inputs. See the generator docstring for the derivation and limits.
"""
from math import fsum as _fsum
''']
    for kind, generator in (("race", _race), ("dp", _dp)):
        for m in range(1, 8):
            for last in range(m):
                pieces.append(generator(m, last))
        table = ["()"] + ["(" + ", ".join(f"_{kind}_{m}_{r}" for r in range(m)) + ",)" for m in range(1, 8)]
        pieces.append(f"_{kind.upper()} = ({', '.join(table)})")
    pieces.append('''def fast_value(hero_stack, others_tuple, payout_differences, last_relevant):
    """Expected ICM payout in excess of last-place base for a positive hero.

    others_tuple contains only positive opponent stacks, at most seven.
    payout_differences[i] is payout[i]-base; last_relevant is the last nonzero
    index (0..len(others_tuple)-1), or -1 for equal payouts. Inputs are trusted,
    matching the validated caller's domain. Returns 0 for no opponents/equal
    payouts. Do not pass an eliminated hero: its finishing payout belongs in
    the caller. Floating-point precision is tested for integer-chip tables.
    """
    if last_relevant < 0 or not others_tuple:
        return 0.0
    return _RACE[len(others_tuple)][last_relevant](hero_stack, others_tuple, payout_differences)


def fast_value_dp(hero_stack, others_tuple, payout_differences, last_relevant):
    """Positive-term unrolled DP alternative with the same API."""
    if last_relevant < 0 or not others_tuple:
        return 0.0
    return _DP[len(others_tuple)][last_relevant](hero_stack, others_tuple, payout_differences)
''')
    return "\n\n".join(pieces) + "\n"


def benchmark(repeats=3000):
    from policy.icm import value as reference
    from policy.icm_fast import fast_value, fast_value_dp

    def adapt(kernel):
        def implementation(stacks, hero, payouts):
            if stacks[hero] <= 0:
                return float(payouts[len(stacks) - 1])
            others = tuple(s for i, s in enumerate(stacks) if i != hero and s > 0)
            if not others:
                return float(payouts[0])
            base = payouts[len(others)]
            differences = tuple(p - base for p in payouts[:len(others)])
            last = max((i for i, p in enumerate(differences) if p), default=-1)
            return base + kernel(stacks[hero], others, differences, last)
        return implementation

    methods = {"current_loop": reference, "unrolled_dp": adapt(fast_value_dp),
               "subset_moments": adapt(fast_value)}
    rng = random.Random(180075)
    states = [[rng.randrange(1, 16001) for _ in range(8)] for _ in range(64)]
    profiles = {"graded8": [100, 70, 50, 35, 24, 15, 8, 0],
                "top3": [50, 30, 20, 0, 0, 0, 0, 0],
                "flat4": [25, 25, 25, 25, 0, 0, 0, 0],
                "winner": [100, 0, 0, 0, 0, 0, 0, 0]}
    results = {}
    max_error = {"unrolled_dp": 0.0, "subset_moments": 0.0}

    def summarize(samples):
        ordered = sorted(samples)
        return {"n": len(samples), "p50_us": statistics.median(samples) / 1000,
                "p95_us": ordered[int(.95 * (len(samples) - 1))] / 1000,
                "p99_us": ordered[int(.99 * (len(samples) - 1))] / 1000,
                "max_us": ordered[-1] / 1000,
                "over_1ms": sum(s > 1_000_000 for s in samples)}

    for profile, payouts in profiles.items():
        samples = {name: [] for name in methods}
        for i in range(repeats):
            stacks, hero = states[i % len(states)], i % 8
            values = {}
            names = list(methods)
            if i % 2:
                names.reverse()
            for name in names:
                start = perf_counter_ns()
                values[name] = methods[name](stacks, hero, payouts)
                samples[name].append(perf_counter_ns() - start)
            for name in max_error:
                max_error[name] = max(max_error[name], abs(values[name] - values["current_loop"]))
        results[profile] = {name: summarize(sample) for name, sample in samples.items()}

    # The three real counterfactual valuations dominate the call-risk branch.
    # Measure all three plus fold/win/loss construction and threshold arithmetic.
    stacks = [1900, 900, 1800, 500, 2400, 800, 1100, 1400]
    payouts = profiles["graded8"]
    risk_samples = {}
    risk_results = {}
    for name, function in methods.items():
        samples = []
        for _ in range(repeats):
            start = perf_counter_ns()
            fold = stacks.copy(); fold[1] += 1200
            win = stacks.copy(); win[0] += 1200
            lose = stacks.copy(); lose[0] -= 800; lose[1] += 2000
            ef, ew, el = function(fold, 0, payouts), function(win, 0, payouts), function(lose, 0, payouts)
            threshold = min(1.0, max(0.0, (ef - el) / (ew - el)))
            samples.append(perf_counter_ns() - start)
        risk_samples[name] = summarize(samples)
        risk_results[name] = threshold
    root = Path(__file__).resolve().parent.parent
    return {"kind": "LOCAL_EXACT_ICM_MICROBENCHMARK", "python": platform.python_version(),
            "platform": platform.platform(), "seed": 180075, "repeats": repeats,
            "source_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                              for path in ("policy/icm.py", "policy/icm_fast.py", "training/compile_icm.py")},
            "value_latency": results, "three_valuation_call_risk": risk_samples,
            "call_threshold_values": risk_results, "max_abs_error_vs_loop": max_error,
            "notes": ["Wall-clock samples include scheduler noise; no timing guarantee.",
                      "Adapters include tuple construction and payout differences; no caller source changed.",
                      "Subset moments are algebraically exact; floating-point roundoff remains."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "policy" / "icm_fast.py")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repeats", type=int, default=3000)
    parser.add_argument("--metrics", type=Path, default=Path(__file__).resolve().parent / "icm_fast_metrics.json")
    args = parser.parse_args()
    code = render()
    compile(code, str(args.out), "exec")
    args.out.write_text(code, encoding="utf-8")
    print(f"Generated {args.out.name}: {len(code.encode('utf-8')):,} bytes, 56 static kernels.")
    if args.benchmark:
        metrics = benchmark(args.repeats)
        args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps({"three_valuation_call_risk": metrics["three_valuation_call_risk"],
                          "max_abs_error_vs_loop": metrics["max_abs_error_vs_loop"]}, indent=2))


if __name__ == "__main__":
    main()
