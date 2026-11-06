"""Generate reproducible preflop asset using ONLY our independently authored code.

Each class has five/n configurable narrowing levels and 1..7 opponents.
Atomic checkpoint after each class. Run in bounded compute resources.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import equity

def generate_class(task):
    index,hole,ranges,samples,seed=task
    values=[[round(equity.native_equity(hole,opponents=n,range_strength=tight,
        seed=seed+index*971+ri*37+n,samples=samples),7) for n in range(1,8)] for ri,tight in enumerate(ranges)]
    return equity.hand_class(hole),values,equity.sampler_stats(reset=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=int,default=30000)
    p.add_argument('--seed',type=int,default=170041)
    p.add_argument('--output',type=Path,default=ROOT/'data'/'preflop.json')
    p.add_argument('--ranges',default='0,.35,.60,.80,.90,1')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    ranges=[float(x) for x in args.ranges.split(',')]
    data={'method':'own native Monte Carlo, heuristic top-fraction opponent ranges','samples_per_cell':args.samples,
          'seed':args.seed,'ranges':ranges,'equities':{},'official_data':False}
    if args.resume and args.output.exists():
        data=json.loads(args.output.read_text(encoding='utf-8'))
        if data['samples_per_cell']!=args.samples or data['ranges']!=ranges or data['seed']!=args.seed:
            raise SystemExit('Resume settings do not match existing asset')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not 1 <= args.workers <= 8:
        raise SystemExit('workers must be between 1 and 8')
    start=time.time();index=0;tasks=[]
    for hi in range(12,-1,-1):
        for lo in range(hi,-1,-1):
            variants=[(hi*4,lo*4+1)] if hi==lo else [(hi*4,lo*4),(hi*4,lo*4+1)]
            for hole in variants:
                name=equity.hand_class(hole)
                if name not in data['equities']:
                    tasks.append((index,hole,ranges,args.samples,args.seed))
                index+=1
    def record(result):
        name,values,stats=result
        data['equities'][name]=values
        totals=data.setdefault('sampler_stats',{})
        for key,value in stats.items():totals[key]=totals.get(key,0)+value
        temporary=args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,separators=(',',':'))+'\n',encoding='utf-8')
        for attempt in range(8):
            try:
                temporary.replace(args.output)
                break
            except PermissionError:
                if attempt == 7:raise
                time.sleep(.025*(attempt+1))
        print(f'{len(data["equities"])}/169 {name} elapsed={time.time()-start:.1f}s',flush=True)
    if args.workers == 1:
        for task in tasks:record(generate_class(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(generate_class,task) for task in tasks]
            for future in as_completed(futures):record(future.result())
    print(f'Complete: {args.output}',flush=True)

if __name__=='__main__':main()
