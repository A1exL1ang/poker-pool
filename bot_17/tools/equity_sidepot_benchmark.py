"""Pathological 7-opponent, 7-distinct-sidepot warm equity latency."""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import equity

def main():
    p=argparse.ArgumentParser();p.add_argument('--calls',type=int,default=2000)
    p.add_argument('--output',type=Path,default=ROOT/'data'/'equity_latency_sidepots.json');args=p.parse_args()
    rng=random.Random(173119);cases=[]
    for i in range(args.calls):
        c=rng.sample(range(52),7);cases.append((c[:2],c[2:2+(3,4,5)[i%3]],[(rng.random(),rng.random()) for _ in range(7)]))
    pots=[(100+(n*37), (1<<n)-1) for n in range(1,8)]
    for h,b,ranges in cases[:100]:equity.estimate(h,b,ranges=ranges,pots=pots,samples=64)
    times=[]
    for h,b,ranges in cases:
        start=time.perf_counter_ns();value=equity.estimate(h,b,ranges=ranges,pots=pots,samples=64);times.append((time.perf_counter_ns()-start)/1000)
        assert 0<=value<=1
    times.sort();result={'backend':equity.BACKEND,'calls':args.calls,'pots':7,'opponents':7,'mean_us':statistics.mean(times),
        'p50_us':times[len(times)//2],'p95_us':times[int(len(times)*.95)],'p99_us':times[int(len(times)*.99)],'max_us':max(times),'over_1ms':sum(t>1000 for t in times)}
    print(json.dumps(result,indent=2));args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
