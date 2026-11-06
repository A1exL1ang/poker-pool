"""Report warm equity latency and accuracy for reproducible fixed workloads."""
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
    p=argparse.ArgumentParser();p.add_argument('--calls',type=int,default=3000)
    p.add_argument('--samples',type=int,default=96);p.add_argument('--output',type=Path)
    args=p.parse_args();rng=random.Random(170555);cases=[]
    for i in range(args.calls):
        cards=rng.sample(range(52),7);nb=(3,4,5)[i%3];n=(1,2,3,7)[i%4]
        cases.append((cards[:2],cards[2:2+nb],n,(0,.6,.9,1.)[i%4],(0,.4,.8,1.)[(i//4)%4]))
    for c in cases[:100]:equity.estimate(*c[:3],range_strength=c[3],aggression=c[4],samples=args.samples)
    equity.sampler_stats(reset=True)
    times=[];groups={}
    for i,c in enumerate(cases):
        start=time.perf_counter_ns();value=equity.estimate(*c[:3],range_strength=c[3],aggression=c[4],seed=17000+i,samples=args.samples)
        elapsed=(time.perf_counter_ns()-start)/1000;times.append(elapsed);groups.setdefault(str(c[2]),[]).append(elapsed)
        assert 0<=value<=1
    def summary(v):
        s=sorted(v);return {'n':len(s),'mean_us':statistics.mean(s),'p50_us':s[len(s)//2],
            'p95_us':s[int(len(s)*.95)],'p99_us':s[int(len(s)*.99)],'max_us':max(s),'over_1ms':sum(x>1000 for x in s)}
    result={'backend':equity.BACKEND,'samples':args.samples,'overall':summary(times),'by_opponents':{k:summary(v) for k,v in groups.items()},'sampler_stats':equity.sampler_stats()}
    print(json.dumps(result,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
