"""Exact river stress evaluation on deliberately unusual boards (development).

These hand-picked positions are diagnostic, not representative tournament data.
Any changes based on their errors must be evaluated on a fresh final holdout.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from equity.evaluator import evaluate
from equity.model import estimate, MODEL_STATUS
from equity.preflop import CLASS_ORDER, class_index, RANGE_FRACTIONS


def parse(text):
    return [4*"23456789TJQKA".index(c[0])+"cdhs".index(c[1]) for c in text.split()]


def format_cards(cards):
    return " ".join("23456789TJQKA"[c//4]+"cdhs"[c%4] for c in cards)


def make_ranges():
    all_combos=[(a,b) for a in range(52) for b in range(a+1,52)]
    result={}
    for fraction in RANGE_FRACTIONS:
        selected=[]
        for cls in CLASS_ORDER:
            selected.extend(h for h in all_combos if class_index(h)==cls)
            if len(selected)>=1326*fraction:break
        result[fraction]=selected
    return result


def exact(hole,board,combos):
    blocked=set((*hole,*board)); hero=evaluate((*hole,*board));total=0;win=0
    for a,b in combos:
        if a in blocked or b in blocked:continue
        value=evaluate((*board,a,b));total+=1;win+=(hero>value)+.5*(hero==value)
    return win/total


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="training/stress.json");args=parser.parse_args()
    ranges=make_ranges();rng=random.Random(180441)
    boards=[("board_quads","Ac Ad Ah As 2c"),("board_quads","2c 2d 2h 2s Ac"),
            ("board_quads","7c 7d 7h 7s 9c"),("board_full_house","Ac Ad Ah Kc Kd"),
            ("board_full_house","2c 2d 2h 3c 3d"),("board_trips","7c 7d 7h Ac 2d"),
            ("board_trips","Ac Ad Ah 7c 2d"),("four_flush","Ac Kc 8c 3c 2d"),
            ("four_flush","2c 4c 7c 9c Ah"),("five_flush","2c 4c 7c 9c Ac"),
            ("four_straight","4c 5d 6h 7s Kc"),("four_straight","Ac 2d 3h 4s Kc"),
            ("board_straight","Ac Kd Qh Js Tc"),("board_straight","Ac Kc Qc Js Td"),
            ("board_straight","4c 5d 6h 7s 8c"),("two_pair","Ac Ad Kc Kd 2h"),
            ("two_pair","2c 2d 3c 3d Ah"),("one_pair","Ac Ad Kh Qs Jh")]
    rows=[]
    for kind,text in boards:
        board=parse(text);remaining=[c for c in range(52) if c not in board]
        for _ in range(6):
            hole=rng.sample(remaining,2)
            for fraction in (.06,.25,1.0):
                target=exact(hole,board,ranges[fraction]);prediction=estimate(hole,board,fraction,1)
                rows.append({"kind":kind,"hole":format_cards(hole),"board":text,"range":fraction,
                             "equity":target,"prediction":prediction,"error":prediction-target})
    report={"purpose":"development stress diagnostics, not representative final evaluation",
            "model_status":MODEL_STATUS,"rows":len(rows),
            "mae":statistics.mean(abs(r["error"]) for r in rows),
            "by_kind":{},"worst":sorted(rows,key=lambda r:abs(r["error"]),reverse=True)[:40]}
    for kind in sorted({r["kind"] for r in rows}):
        group=[r for r in rows if r["kind"]==kind]
        report["by_kind"][kind]={"n":len(group),"mae":statistics.mean(abs(r["error"]) for r in group),"max_abs":max(abs(r["error"]) for r in group)}
    Path(args.output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
