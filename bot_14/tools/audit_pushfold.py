"""Audit a locked shove/fold policy against an independent equity matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.train_pushfold import audit, conditional_counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", default="assets/pushfold.json")
    p.add_argument("--equity", required=True)
    p.add_argument("--output", default="measurements/pushfold_heldout.json")
    args = p.parse_args()
    import numpy as np
    policy = json.loads(Path(args.policy).read_text())
    data = json.loads(Path(args.equity).read_text())
    counts, combos = conditional_counts()
    conditional = np.asarray(counts, dtype=float) / 1225
    prior = np.asarray(combos, dtype=float) / 1326
    e = np.asarray(data["equity"], dtype=float)
    depth = np.asarray(policy["depths_bb"], dtype=float)[:, None, None]
    a = conditional[None, :, :] * ((2 * e[None, :, :] - 1) * depth - 1)
    b = conditional[None, :, :] * ((2 * e[None, :, :] - 1) * depth + 1)
    reports = audit(np, prior, conditional, a, b, np.asarray(policy["sb_shove"]), np.asarray(policy["bb_call"]))
    result = dict(phase="independent equity audit of locked heads-up shove/fold abstraction", policy_sha256=hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(), equity_sha256=hashlib.sha256(Path(args.equity).read_bytes()).hexdigest(), independent_equity_seed=data["seed"], trials_per_pair=data["trials_per_unordered_class_pair"], audits=reports, maximum_nash_conv_gap_bb=max(r["nash_conv_gap_bb"] for r in reports), limitations="Monte Carlo matrix uncertainty remains; this is not exploitability in full NLHE or multiplayer tournaments.")
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
