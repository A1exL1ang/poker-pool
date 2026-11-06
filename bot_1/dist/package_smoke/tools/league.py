"""Seeded tournaments against deliberately simple original test controls."""
import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pokerbot.agent import Agent
from pokerbot.simulator import run_sng
from pokerbot.evaluator import parse_card, evaluate
from pokerbot.equity import hand_class, STRENGTH


class Control:
    """Weak test opponents, not competition submissions or other teams' code."""
    def __init__(self, style, seed):
        self.style=style
        self.rng=random.Random(seed)

    def act(self,o):
        specs={x["action"]:x for x in o["legal_actions"]}
        me=next(p for p in o["players"] if p["seat"]==o["hero"])
        cost=o["to_call"]
        hole=[parse_card(c) for c in o["hole_cards"]]
        board=o["board"]
        strength=STRENGTH[hand_class(*hole)] if not board else 0
        category=evaluate(hole+board)>>24 if board else 0
        r=self.rng.random()
        raise_it=False
        call_it=False
        if self.style=="station":
            call_it=not board or category>=1 or cost<o["pot"]*.25
            raise_it=category>=3 or not board and strength>.79
        elif self.style=="nit":
            call_it=strength>.63 if not board else category>=2
            raise_it=strength>.75 if not board else category>=3
        elif self.style=="loose":
            call_it=strength>.43 if not board else category>=1 or r<.24
            raise_it=strength>.67 if not board else category>=2 or r<.12
        elif self.style=="maniac":
            call_it=r<.82
            raise_it=r<.38
        elif self.style=="rock":
            call_it=strength>.59 if not board else category>=1
            raise_it=strength>.69 if not board else category>=2
        elif self.style=="random":
            call_it=r<.55
            raise_it=r<.12
        if raise_it and "raise" in specs:
            spec=specs["raise"]
            size=max(o["big_blind"]*3,int(o["pot"]*.7)+me["bet"]+cost)
            return {"action":"raise","amount":max(spec["min"],min(spec["max"],size))}
        if "check" in specs:
            return {"action":"check"}
        if call_it:
            return {"action":"call","amount":cost}
        return {"action":"fold"}


def league(tables=100, seed=71, lineup="mixed", selfplay=False):
    started=time.perf_counter()
    scores=[]
    wins=0
    chips=[]
    actions=hands=illegal=0
    for i in range(tables):
        hero_seat=i%8
        styles=["station","nit","loose","maniac","rock","random","loose"] if lineup=="mixed" else [lineup]*7
        policies=[]
        j=0
        for s in range(8):
            if s==hero_seat or selfplay:
                policies.append(Agent(seed+i*97+s))
            else:
                policies.append(Control(styles[j],seed+i*97+s))
                j+=1
        t,points=run_sng(policies,seed+i*131)
        scores.append(points[hero_seat])
        chips.append(t.players[hero_seat].stack)
        wins+=t.players[hero_seat].stack==16000
        hands+=t.hand_id
        actions+=t.decisions
        illegal+=t.illegal
    mean=statistics.mean(scores)
    se=statistics.stdev(scores)/math.sqrt(tables) if tables>1 else 0
    return {"tables":tables,"seed":seed,"lineup":"selfplay" if selfplay else lineup,"fixture_payouts":[50,30,20,0,0,0,0,0],"fixture_blinds":[20,40,80,120,200,400,800,1600],"hands_per_level":25,"hand_limit":200,"equal_share_baseline":12.5,"mean_points":mean,"standard_error":se,"normal_95_interval":[mean-1.96*se,mean+1.96*se],"outright_wins":wins,"mean_terminal_chips":statistics.mean(chips),"hands":hands,"decisions":actions,"illegal_actions":illegal,"elapsed_seconds":time.perf_counter()-started,"limitations":"Synthetic opponents/schedule/payouts; not evidence of official contest win probability. Normal interval is descriptive for these independent fixture tables."}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--tables",type=int,default=100)
    p.add_argument("--seed",type=int,default=71)
    p.add_argument("--lineup",default="mixed",choices=["mixed","station","nit","loose","maniac","rock","random"])
    p.add_argument("--selfplay",action="store_true")
    p.add_argument("--output",default="benchmarks/league.json")
    a=p.parse_args()
    result=league(a.tables,a.seed,a.lineup,a.selfplay)
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
