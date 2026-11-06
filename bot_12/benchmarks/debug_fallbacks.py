"""Reproduce rare observed internal fallbacks using own validation seeds."""
import json
from pathlib import Path
import random
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent import Agent
from baselines import make_lineup
from simulator import run_tournament

EVENTS=[]


class DebugAgent(Agent):
    def act(self,o):
        self.decisions+=1
        try:
            self._observe(o)
            a=self._preflop(o) if o['street']=='preflop' else self._postflop(o)
            return self._legal(o,a)
        except Exception:
            self.fallbacks+=1
            event={'observation':o,'params':self.params,'traceback':traceback.format_exc()}
            EVENTS.append(event)
            print(event['traceback'],flush=True)
            (ROOT/'.control/fallback_debug.json').write_text(json.dumps(EVENTS,indent=2))
            return {'action':'check' if 'check' in o['legal_actions'] else 'fold'}


def main():
    for name in ('cautious','tight_defend'):
        params=json.loads((ROOT/'measurements/params'/f'{name}.json').read_text())
        for index in range(512):
            seed=110000+index
            hero=DebugAgent(seed=seed^120012,params=params)
            rivals=make_lineup('selfplay',seed*17)
            random.Random(seed^0x12B0).shuffle(rivals)
            rivals.insert(index%8,hero)
            r=run_tournament(rivals,seed)
            if hero.fallbacks:
                print(json.dumps({'name':name,'index':index,'seed':seed,'fallbacks':hero.fallbacks}),flush=True)
        print('completed '+name,flush=True)


if __name__=='__main__':
    main()
