"""Seed-blocked, seat-balanced frozen-bot evaluation. One JSONL per shard."""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from adapters import Adapter, Baseline
from common.engine import Config,run_tournament

PAYOUTS={'top3':(.5,.3,.2,0,0,0,0,0),'winner':(1,0,0,0,0,0,0,0),'top4':(.4,.3,.2,.1,0,0,0,0)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--seeds',type=int,default=64);p.add_argument('--output',required=True)
    p.add_argument('--pilot',action='store_true');a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if out.exists():
        for line in out.read_text().splitlines(): done.add(json.loads(line)['match_id'])
    count=0
    with out.open('a',buffering=1) as f:
        for payout,pay in PAYOUTS.items():
            if a.pilot and payout!='top3':continue
            for seed_index in range(a.seeds):
                # All nine leave-one-out fields: omit 0 => all our eight.
                for omitted in range(9):
                    field=[i for i in range(9) if i!=omitted]
                    for rotation in range(8):
                        index=count;count+=1
                        if index%a.shards!=a.shard:continue
                        mid=f'{payout}:{seed_index}:{omitted}:{rotation}'
                        if mid in done:continue
                        seats=field[rotation:]+field[:rotation]
                        bots=[Baseline() if i==0 else Adapter(i) for i in seats]
                        seed=2026110600+seed_index
                        start=time.perf_counter()
                        r=run_tournament(bots,Config(payouts=pay,strict_timing=True),seed)
                        row=dict(match_id=mid,payout=payout,seed_index=seed_index,seed=seed,omitted=omitted,rotation=rotation,seats=seats,elapsed_s=time.perf_counter()-start,**dataclasses.asdict(r))
                        row.pop('hand_records')
                        row['internal_errors']=[getattr(b.bot,'errors',0)+getattr(b,'internal_errors',0) for b in bots]
                        f.write(json.dumps(row,separators=(',',':'))+'\n')
    print(json.dumps({'shard':a.shard,'output':str(out),'complete':True}))

if __name__=='__main__':main()
