"""Diagnostic after unexpected self-play sample; frozen policy is unchanged.

Checks that focal-first versus natural Agent construction does not change a
seeded all-mirror game. Uses fresh diagnostic seeds, timing disabled, and keeps
the original held-out sample intact. This is a state-isolation check, not tuning.
"""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bot import Agent
from engine import tournament
from evaluator import evaluate
from tools.benchmark import SYNTHETIC_PAYOUTS,SYNTHETIC_BLINDS


def main():
    records=[]
    for seed in (230001,230009,230027,230041):
        games=[]
        for first in (0,5):
            agents=[None]*8
            order=[first]+[i for i in range(8) if i!=first]
            for seat in order:
                agents[seat]=Agent(seed=seed*101+seat+9)
            r=tournament(agents,seed=seed,payouts=SYNTHETIC_PAYOUTS['sng'],
                blind_schedule=SYNTHETIC_BLINDS,evaluator=evaluate,
                enforce_time_limit=False,record_latencies=False)
            games.append({k:r[k] for k in ('stacks','points','hands','actions_by_seat',
                'errors','decisions','finish_probabilities')})
        assert games[0]==games[1],(seed,games)
        assert abs(sum(games[0]['points'])/8-.125)<1e-12
        records.append({'seed':seed,'construction_orders_identical':True,
            'points':games[0]['points'],'all_seat_mean':sum(games[0]['points'])/8,
            'decisions':games[0]['decisions']})
    report={'purpose':'state-isolation diagnostic; no policy tuning; original self-play result retained',
        'policy_sha256':hashlib.sha256(Path('bot.py').read_bytes()).hexdigest(),
        'records':records,'failures':0,
        'limitation':'Does not prove absence of every benchmark or strategy error; self-play sample variation remains.'}
    Path('measurements/selfplay_construction_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
