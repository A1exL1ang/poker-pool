"""Randomized interface/legality checks, including deep/short stacks and all-ins."""
import argparse
import json
import random
from pathlib import Path
from collections import Counter
from agent import Agent,Observation,Player
from tools.simulator import action_is_legal


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--states',type=int,default=20000);args=parser.parse_args()
    rng=random.Random(818282);bot=Agent();actions=Counter();streets=Counter()
    for i in range(args.states):
        n=rng.randint(2,8);hero=rng.randrange(n);button=rng.randrange(n);bb=rng.choice([2,20,100,1000])
        board_n=rng.choice([0,3,4,5]);cards=rng.sample(range(52),board_n+2)
        current=rng.randint(0,40)*bb;players=[]
        for s in range(n):
            committed=rng.randint(0,current) if current else 0
            invested=committed+(rng.randint(0,30)*bb if board_n else 0)
            stack=rng.choice([0,1,5,20,100])*bb+rng.randint(0,bb)
            if s==hero:stack=max(1,stack)
            folded=s!=hero and stack>0 and rng.random()<.35
            players.append(Player(s,stack,committed,invested,folded,stack==0))
        # Ensure at least one live opponent; all inputs expose public info only.
        villain=(hero+1)%n;p=players[villain]
        players[villain]=Player(p.seat,p.stack,p.committed,p.invested,False,p.all_in)
        h=players[hero];cost=current-h.committed
        o=Observation(i,hero,button,tuple(cards[:2]),tuple(cards[2:]),tuple(players),sum(p.invested for p in players),
                      cost,current+bb,h.stack+h.committed,bb,rng.random()<.8,table_id=f'fuzz-{i//100}')
        o.validate()
        a=bot.act(o)
        if not action_is_legal(o,a):raise AssertionError((o,a))
        actions[a.action]+=1;streets[o.street]+=1
    result={'states':args.states,'seed':818282,'illegal_actions':0,'exceptions':0,'actions':dict(actions),'streets':dict(streets),
            'scope':'Valid local-contract synthetic states; not every generated state is reachable through a legal hand. Reachable-state tests are in tournament simulation.'}
    Path('evidence/stress.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
