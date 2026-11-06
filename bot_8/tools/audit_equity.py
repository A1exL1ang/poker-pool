"""Quantify bounded river-sampling error against exact range enumeration."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path
from kestrel.cards import EMPTY,extend,score
from kestrel.equity import POOLS,estimate,board_context,fast_quality


def exact(hole,board,profile):
    idx,aggression,passive,bluff=profile
    state=extend(EMPTY,board); context=board_context(board)
    hero=score(extend(state,hole)); board_score=score(state)
    dead=set(hole+board); wins=total=0.
    for a,b in POOLS[idx]:
        if a in dead or b in dead:
            continue
        v=score(extend(state,(a,b)))
        strength=fast_quality((a,b),context,v)
        if v==board_score:
            strength=.27
        weight=(bluff+(1-bluff)*strength**(2+aggression)) if aggression else 1.
        if passive:
            weight*=(.28+.72*strength)**passive
        total+=weight
        wins+=weight*(1 if hero>v else .5 if hero==v else 0)
    return wins/total


def main():
    p=argparse.ArgumentParser();p.add_argument('--states',type=int,default=1000);p.add_argument('--output',default='evidence/equity_audit.json')
    args=p.parse_args();rng=random.Random(89023);errors=[];signed=[]
    for i in range(args.states):
        cards=rng.sample(range(52),7);hole=tuple(cards[:2]);board=tuple(cards[2:])
        profile=(rng.choice([2,4,6,7]),rng.choice([0.,1.,2.,3.]),rng.choice([0.,.65]),.10)
        reference=exact(hole,board,profile)
        q,_,_=estimate(hole,board,[profile],i*7001,.5)
        signed.append(q-reference);errors.append(abs(q-reference))
    result={'states':args.states,'seed':89023,'samples':40,'time_guard':False,
            'mean_absolute_error':statistics.fmean(errors),'root_mean_square_error':math.sqrt(statistics.fmean(e*e for e in errors)),
            'mean_signed_error':statistics.fmean(signed),'p95_absolute_error':sorted(errors)[int(.95*(len(errors)-1))],
            'max_absolute_error':max(errors),
            'scope':'River heads-up versus the SAME heuristic range/action likelihood, enumerated exactly. Measures sampling error, not correctness of the opponent model. Bounded gameplay may use fewer samples.'}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
