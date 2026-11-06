"""Bounded randomized legality/chip/score-conservation regression exercise."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import random
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines import RandomLegal
from simulator import HoldemHand,run_tournament


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands",type=int,default=10000)
    parser.add_argument("--tournaments",type=int,default=100)
    parser.add_argument("--seed",type=int,default=31291)
    parser.add_argument("--output",default="measurements/mechanics_fuzz.json")
    args=parser.parse_args()
    if not 0<=args.hands<=1_000_000 or not 0<=args.tournaments<=100_000:
        raise ValueError("fuzz counts must be bounded and nonnegative")
    started=perf_counter()
    rng=random.Random(args.seed)
    bots=[RandomLegal(i+71) for i in range(8)]
    decisions=chips=0
    for _ in range(args.hands):
        n=rng.randint(2,8)
        stacks=[rng.randint(1,10000) for _ in range(n)]
        hand=HoldemHand(stacks,rng.randrange(n),rng.randint(1,10),20,rng=rng)
        result=hand.play(bots[:n])
        assert sum(result["stacks"])==sum(stacks)
        assert not any(result["invalid_actions"])
        assert not any(result["exceptions"])
        decisions+=sum(result["decisions"])
        chips+=sum(stacks)
    for k in range(args.tournaments):
        result=run_tournament([RandomLegal(args.seed+k*8+i) for i in range(8)],seed=args.seed+k,
                              starting_stack=100,max_hands=50,
                              blind_schedule=((5,10),(10,20),(20,40),(40,80),(80,160)),
                              hands_per_level=10,payouts=(5,3,2,1))
        assert sum(result["stacks"])==800
        assert abs(sum(result["points"])-11)<1e-9
        assert not any(result["invalid_actions"])
        assert not any(result["exceptions"])
        decisions+=sum(result["decisions"])
    measured=dict(seed=args.seed,hands=args.hands,tournaments=args.tournaments,
                  decisions=decisions,hand_chip_inputs=chips,
                  invariant_failures=0,invalid_actions=0,exceptions=0,
                  seconds=perf_counter()-started,python=sys.version,platform=platform.platform(),
                  scope="Local randomized mechanics validation, not playing-strength evidence.")
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must stay in this bot workspace")
    output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(measured,indent=2))
    print(json.dumps(measured,indent=2))


if __name__=="__main__":
    main()
