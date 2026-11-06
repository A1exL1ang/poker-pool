"""Offline HU shove/fold CFR+ using independently generated preflop labels.

This is a small, explicitly restricted game: the SB either folds or shoves;
after a shove the BB either folds or calls. There are no antes, limps, small
raises or postflop play. Both players start with ``depth`` big blinds,
including the SB=.5 and BB=1 already posted. The game is zero-sum chip EV,
appropriate to HU ICM because two-player ICM is linear in chips.

Only the offline trainer imports NumPy. Gameplay reads the resulting JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


DEFAULT_DEPTHS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0)


def class_id(a: int, b: int) -> int:
    r0, r1 = a // 4, b // 4
    high, low = max(r0, r1), min(r0, r1)
    if high == low:
        return (12 - high) * 13 + (12 - high)
    return (12 - high) * 13 + (12 - low) if a % 4 == b % 4 else (12 - low) * 13 + (12 - high)


def class_names():
    ranks = "AKQJT98765432"
    return [ranks[i] + ranks[j] if i == j else ranks[i] + ranks[j] + "s" if i < j else ranks[j] + ranks[i] + "o"
            for i in range(13) for j in range(13)]


def conditional_counts():
    """Exact villain-class combo counts after a canonical hero-class hand.

    Every physical combo of a given hero class is equivalent under suit
    permutation, so one representative suffices. Rows sum to C(50,2)=1225.
    ``combo_count[i] * counts[i][j]`` is exactly symmetric, which is also
    checked here. These blocker weights are not the unconditioned 6/4/12
    class weights often used in rough shove/fold approximations.
    """
    classes = [[] for _ in range(169)]
    all_combos = []
    for a in range(52):
        for b in range(a + 1, 52):
            index = class_id(a, b)
            classes[index].append((a, b))
            all_combos.append((a, b, index))
    combo_count = [len(combos) for combos in classes]
    counts = []
    for combos in classes:
        a, b = combos[0]
        row = [0] * 169
        for c, d, index in all_combos:
            if c != a and c != b and d != a and d != b:
                row[index] += 1
        if sum(row) != 1225:
            raise AssertionError("incorrect conditional combo total")
        counts.append(row)
    if sum(combo_count) != 1326:
        raise AssertionError("incorrect total starting combos")
    for i in range(169):
        for j in range(169):
            if combo_count[i] * counts[i][j] != combo_count[j] * counts[j][i]:
                raise AssertionError("asymmetric physical card deal")
    return counts, combo_count


def load_asset(path: Path):
    raw = path.read_bytes()
    data = json.loads(raw)
    names = class_names()
    if data.get("classes") != names:
        raise ValueError("preflop class order must match the canonical A..2 grid")
    equity = data.get("equity")
    if not isinstance(equity, list) or len(equity) != 169 or any(len(row) != 169 for row in equity):
        raise ValueError("expected a 169x169 heads-up equity matrix")
    for i, row in enumerate(equity):
        for j, value in enumerate(row):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid showdown equity")
            if abs(value + equity[j][i] - 1.0) > 2e-8:
                raise ValueError("showdown matrix must obey player-exchange symmetry")
    counts, combo_count = conditional_counts()
    if data.get("combination_counts") != combo_count:
        raise ValueError("preflop combination counts disagree with independently enumerated counts")
    return data, counts, combo_count, hashlib.sha256(raw).hexdigest()


def toy_validation():
    """Small pure-Python checks of payoff signs and regret updates.

    Equal hand equity has a simple pure equilibrium at every depth: both
    shove and call. This validates blind accounting independently of NumPy.
    A second asymmetric two-type game checks zero-sum payoff accounting.
    """
    reports = []
    for depth in (1, 2, 10, 25):
        r_sf = r_ss = r_bf = r_bc = 0.0
        avg_s = avg_c = weight = 0.0
        for iteration in range(1, 201):
            s = r_ss / (r_sf + r_ss) if r_sf + r_ss else 0.5
            c = r_bc / (r_bf + r_bc) if r_bf + r_bc else 0.5
            adv_s = 1.5 - c
            r_sf, r_ss = max(0.0, r_sf - s * adv_s), max(0.0, r_ss + (1 - s) * adv_s)
            s = r_ss / (r_sf + r_ss) if r_sf + r_ss else 0.5
            adv_c = s
            r_bf, r_bc = max(0.0, r_bf - c * adv_c), max(0.0, r_bc + (1 - c) * adv_c)
            c = r_bc / (r_bf + r_bc) if r_bf + r_bc else 0.5
            avg_s += iteration * s
            avg_c += iteration * c
            weight += iteration
        s, c = avg_s / weight, avg_c / weight
        sb_value = -.5 + s * (1.5 - c)
        bb_value = .5 - 1.5 * s + c * s
        gap = (1 - s) * (1.5 - c) + (1 - c) * s
        if abs(sb_value + bb_value) > 1e-12 or gap > 1e-12 or s != 1 or c != 1:
            raise AssertionError("equal-equity toy equilibrium failed")
        reports.append({"depth": depth, "sb_shove": s, "bb_call": c, "gap_bb": gap})
    e = [[.5, .8], [.2, .5]]
    s, c, depth = [.3, .7], [.2, .9], 6
    sb = bb = 0.0
    for i in range(2):
        for j in range(2):
            called = (2 * e[i][j] - 1) * depth
            sb += .25 * ((1 - s[i]) * -.5 + s[i] * ((1 - c[j]) * 1 + c[j] * called))
            bb += .25 * ((1 - s[i]) * .5 + s[i] * ((1 - c[j]) * -1 - c[j] * called))
    if abs(sb + bb) > 1e-12:
        raise AssertionError("asymmetric toy zero-sum accounting failed")
    return reports


def audit(np, prior, conditional, a_kernel, b_kernel, shove, call):
    """Exact best-response audit in the 169-class shove/fold abstraction."""
    adv_s = 1.5 + np.einsum("dij,dj->di", a_kernel, call, optimize=False)
    adv_c = np.einsum("dij,dj->di", b_kernel, shove, optimize=False)
    value = ((-.5 + shove * adv_s) * prior).sum(axis=1)
    reach = shove @ conditional.T
    bb_value = ((.5 - 1.5 * reach + call * adv_c) * prior).sum(axis=1)
    if np.max(np.abs(value + bb_value)) > 1e-8:
        raise AssertionError("nonzero-sum blocker/payoff accounting")
    sb_gain = ((np.maximum(adv_s, 0) - shove * adv_s) * prior).sum(axis=1)
    bb_gain = ((np.maximum(adv_c, 0) - call * adv_c) * prior).sum(axis=1)
    gap = np.maximum(0, sb_gain + bb_gain)
    reports = []
    for d in range(len(shove)):
        reports.append({
            "sb_value_bb": float(value[d]),
            "bb_value_bb": float(bb_value[d]),
            "sb_best_response_value_bb": float(value[d] + sb_gain[d]),
            "bb_best_response_value_bb": float(-value[d] + bb_gain[d]),
            "sb_best_response_gain_bb": float(max(0, sb_gain[d])),
            "bb_best_response_gain_bb": float(max(0, bb_gain[d])),
            "nash_conv_gap_bb": float(gap[d]),
            "exploitability_half_gap_bb": float(gap[d] / 2),
            "sb_shove_frequency": float(shove[d] @ prior),
            "bb_call_frequency_if_faced_independent_hand": float(call[d] @ prior),
            "bb_call_frequency_given_shove": float((call[d] * reach[d]) @ prior / max(1e-30, reach[d] @ prior)),
        })
    return reports


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def train(args, asset, counts, combo_count, asset_hash):
    # Restrict this trainer's BLAS/OpenMP threads before NumPy is imported.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    import numpy as np

    depths = np.asarray(args.depths, dtype=np.float64)
    conditional = np.asarray(counts, dtype=np.float64) / 1225
    prior = np.asarray(combo_count, dtype=np.float64) / 1326
    equity = np.asarray(asset["equity"], dtype=np.float64)
    a_kernel = conditional[None, :, :] * ((2 * equity[None, :, :] - 1) * depths[:, None, None] - 1)
    b_kernel = conditional[None, :, :] * ((2 * equity[None, :, :] - 1) * depths[:, None, None] + 1)
    shape = (len(depths), 169)
    regret_sf, regret_ss, regret_bf, regret_bc = [np.zeros(shape) for _ in range(4)]
    sum_s, sum_c = np.zeros(shape), np.zeros(shape)
    weight_sum = 0.0
    iteration_start = 0
    convergence = [[] for _ in depths]
    best_gap = [float("inf")] * len(depths)
    best_s, best_c = np.full(shape, .5), np.full(shape, .5)
    selected_iterations = [0] * len(depths)
    if args.resume:
        resume = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        if resume["preflop_sha256"] != asset_hash or resume["depths_bb"] != args.depths:
            raise ValueError("checkpoint assets or depths do not match")
        if resume.get("average_delay", 0) != args.average_delay:
            raise ValueError("checkpoint averaging delay does not match")
        iteration_start = resume["iteration"]
        regret_sf, regret_ss, regret_bf, regret_bc = [np.asarray(resume[key], dtype=np.float64) for key in ("regret_sf", "regret_ss", "regret_bf", "regret_bc")]
        sum_s, sum_c = [np.asarray(resume[key], dtype=np.float64) for key in ("sum_s", "sum_c")]
        weight_sum = resume["weight_sum"]
        convergence = resume["convergence"]
        best_s, best_c = np.asarray(resume["best_s"]), np.asarray(resume["best_c"])
        best_gap = resume["best_gap"]
        selected_iterations = resume["selected_iterations"]
    if args.iterations <= iteration_start:
        raise ValueError("iterations must exceed the checkpoint iteration")

    def strategy(r0, r1):
        total = r0 + r1
        return np.divide(r1, total, out=np.full_like(total, .5), where=total > 0)

    def update(r0, r1, action, advantage):
        r0 -= action * advantage
        r1 += (1 - action) * advantage
        np.maximum(r0, 0, out=r0)
        np.maximum(r1, 0, out=r1)

    started = time.perf_counter()
    for iteration in range(iteration_start + 1, args.iterations + 1):
        shove = strategy(regret_sf, regret_ss)
        call = strategy(regret_bf, regret_bc)
        adv_s = 1.5 + np.einsum("dij,dj->di", a_kernel, call, optimize=False)
        update(regret_sf, regret_ss, shove, adv_s)
        shove = strategy(regret_sf, regret_ss)
        adv_c = np.einsum("dij,dj->di", b_kernel, shove, optimize=False)
        update(regret_bf, regret_bc, call, adv_c)
        call = strategy(regret_bf, regret_bc)
        weight = max(0, iteration - args.average_delay)
        sum_s += weight * shove
        sum_c += weight * call
        weight_sum += weight
        check = iteration in (1, 10, 50, 100, 250, 500) or iteration % args.checkpoint_every == 0 or iteration == args.iterations
        if not check or weight_sum == 0:
            continue
        average_s, average_c = sum_s / weight_sum, sum_c / weight_sum
        reports = audit(np, prior, conditional, a_kernel, b_kernel, average_s, average_c)
        for d, report in enumerate(reports):
            convergence[d].append({"iteration": iteration, "nash_conv_gap_bb": report["nash_conv_gap_bb"], "sb_value_bb": report["sb_value_bb"]})
            if report["nash_conv_gap_bb"] < best_gap[d]:
                best_gap[d] = report["nash_conv_gap_bb"]
                best_s[d] = average_s[d]
                best_c[d] = average_c[d]
                selected_iterations[d] = iteration
        print(json.dumps({"iteration": iteration, "elapsed_seconds": round(time.perf_counter() - started, 3), "max_gap_bb": max(report["nash_conv_gap_bb"] for report in reports), "mean_gap_bb": sum(report["nash_conv_gap_bb"] for report in reports) / len(reports)}), flush=True)
        checkpoint = {
            "version": 1, "iteration": iteration, "preflop_sha256": asset_hash, "depths_bb": args.depths,
            "average_delay": args.average_delay, "weight_sum": weight_sum,
            "regret_sf": regret_sf.tolist(), "regret_ss": regret_ss.tolist(),
            "regret_bf": regret_bf.tolist(), "regret_bc": regret_bc.tolist(),
            "sum_s": sum_s.tolist(), "sum_c": sum_c.tolist(),
            "best_s": best_s.tolist(), "best_c": best_c.tolist(), "best_gap": best_gap,
            "selected_iterations": selected_iterations, "convergence": convergence,
        }
        atomic_json(Path(args.checkpoint), checkpoint)
        # Keep a runnable best asset throughout long training runs.
        selected_audit = audit(np, prior, conditional, a_kernel, b_kernel, best_s, best_c)
        result = {
            "version": 1,
            "game": "HU SB shove/fold vs BB call/fold; no ante; chip EV",
            "depth_semantics": "equal effective starting stacks including posted blinds; SB=0.5 BB=1",
            "limitations": "restricted action game; Monte Carlo showdown labels; no limps/raises/postflop/multiway/ICM risk premium",
            "algorithm": "alternating CFR+; positive cumulative counterfactual regrets; linearly weighted average strategies",
            "average_delay": args.average_delay,
            "classes": asset["classes"], "depths_bb": args.depths,
            "sb_shove": best_s.tolist(), "bb_call": best_c.tolist(),
            "iterations": iteration, "selected_iterations": selected_iterations,
            "preflop_sha256": asset_hash,
            "preflop_trials_per_unordered_class_pair": asset.get("trials_per_unordered_class_pair"),
            "conditional_combo_total": 1225, "initial_combo_total": 1326,
            "audit": selected_audit, "convergence": convergence,
            "training_elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(Path(args.output), result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", default="assets/preflop.json")
    parser.add_argument("--output", default="assets/pushfold.json")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--depths", type=lambda text: [float(v) for v in text.split(",")], default=list(DEFAULT_DEPTHS))
    parser.add_argument("--average-delay", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint", default=".control/pushfold_checkpoint.json")
    parser.add_argument("--resume")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.average_delay < 0 or args.average_delay >= args.iterations or args.checkpoint_every < 1:
        parser.error("positive iterations/checkpoint period and 0 <= delay < iterations required")
    if not args.depths or any(not math.isfinite(d) or d < 1 for d in args.depths) or args.depths != sorted(set(args.depths)):
        parser.error("depths must be unique ascending values at least 1 BB")
    return args


def main():
    args = parse_args()
    asset, counts, combo_count, asset_hash = load_asset(Path(args.assets))
    toy_reports = toy_validation()
    if args.validate_only:
        print(json.dumps({"classes": len(combo_count), "starting_combos": sum(combo_count), "conditional_combos": sorted(set(map(sum, counts))), "detailed_balance_exact": True, "preflop_sha256": asset_hash, "equal_equity_toy": toy_reports}, indent=2))
        return
    result = train(args, asset, counts, combo_count, asset_hash)
    print(json.dumps({"output": args.output, "iterations": result["iterations"], "max_selected_gap_bb": max(item["nash_conv_gap_bb"] for item in result["audit"])}))


if __name__ == "__main__":
    main()
