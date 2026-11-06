"""Measure warmed feature+equity inference; no scientific runtime dependency."""
import argparse
import json
from pathlib import Path
import platform
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from equity import analyze,USE_EXACT_RIVER,_MODEL

def main():
    p=argparse.ArgumentParser();p.add_argument('--calls',type=int,default=10000)
    p.add_argument('--output',default='training/equity_latency.json');a=p.parse_args()
    r=random.Random(1212881)
    hands=[r.sample(range(52),5+i%3) for i in range(a.calls)]
    for cs in hands[:100]:analyze(cs[:2],cs[2:])
    ts=[]
    for cs in hands:
        hole,board=cs[:2],cs[2:]
        t=time.perf_counter_ns();analyze(hole,board);ts.append((time.perf_counter_ns()-t)/1000)
    st=sorted(ts)
    q=lambda p:st[min(len(st)-1,int(p*len(st))) ]
    out={'python':sys.version,'platform':platform.platform(),'calls':len(ts),'scope':'warmed analyze(hole,board), feature extraction plus both HU model targets',
         'exact_river_enabled':USE_EXACT_RIVER,'forest_algorithm':_MODEL.get('algorithm','separate_HistGradientBoosting') if _MODEL else None,
         'us_mean':sum(ts)/len(ts),'us_p50':q(.5),'us_p95':q(.95),'us_p99':q(.99),'us_p999':q(.999),'us_max':max(ts),
         'over_1ms':sum(t>1000 for t in ts),'fallbacks':0,
         'notes':'Includes OS scheduling jitter. No timer-abort fallback is used in equity inference; full agent timing is separate.'}
    Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
