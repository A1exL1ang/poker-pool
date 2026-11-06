"""Reproducibly generate bot13's own 169-class preflop equity table offline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
from build import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100000, help="rollouts per class and opponent profile")
    parser.add_argument("--seed", type=int, default=130613)
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "preflop.json")
    args = parser.parse_args()
    exe = build(trainer=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    with temp.open("wb") as stream:
        subprocess.run([str(exe), str(args.samples), str(args.seed)], stdout=stream, check=True, cwd=ROOT)
    data = json.loads(temp.read_text())
    if len(data["hands"]) != 169 or any(len(v["random"]) != 7 for v in data["hands"].values()):
        raise RuntimeError("invalid generated table")
    temp.replace(args.output)
    print("Saved", args.output, "samples per equity:", args.samples)


if __name__ == "__main__":
    main()
