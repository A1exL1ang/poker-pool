"""Stdlib deployment audit: native feature parity, label errors and latency.

Usage: python training/equity_audit.py --data artifacts/equity_test.bin
This audit reports the trained selected model without retuning on the test data.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from bot.equity import features,estimate_all,_TREES


def percentile(xs,p):
    return sorted(xs)[min(len(xs)-1,int(p*(len(xs)-1)))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='artifacts/equity_runtime_metrics.json');args=p.parse_args()
    if _TREES is None:raise SystemExit('Trained model missing; refusing to audit developmental fallback.')
    with open(args.data,'rb') as f:
        n,nf=struct.unpack('<ii',f.read(8));rows=list(struct.iter_unpack('<'+str(nf+4)+'f7i',f.read()))
    assert n==len(rows)
    parity=0;errors=[];durations=[];by_street={3:[],4:[],5:[]}
    cases=[]
    for row in rows:
        h=list(row[nf+4:nf+6]);b=[c for c in row[nf+6:] if c>=0];x=features(h,b)
        parity+=x!=list(row[:nf]);cases.append((h,b,row[nf:nf+4]))
    for h,b,y in cases:
        begin=time.perf_counter_ns();pred=estimate_all(h,b);durations.append((time.perf_counter_ns()-begin)/1000)
        e=[a-b for a,b in zip(pred,y)];errors.extend(e);by_street[len(b)].extend(e)
    out={'n':n,'feature_parity_mismatches':parity,'model_trees':len(_TREES),'mae':statistics.mean(map(abs,errors)),
         'rmse':math.sqrt(statistics.mean(e*e for e in errors)),
         'p95_absolute_error':percentile(list(map(abs,errors)),.95),
         'mae_by_street':{k:statistics.mean(map(abs,v)) for k,v in by_street.items()},
         'inference_us':{'median':statistics.median(durations),'p95':percentile(durations,.95),'p99':percentile(durations,.99),'max':max(durations),'over_1000us':sum(t>1000 for t in durations)},
         'notes':'Wall-clock inference includes Python features and all four outputs, excludes import/model loading. Timing is machine/load dependent; errors include Monte Carlo label noise.'}
    Path(args.out).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));assert not parity


if __name__=='__main__':main()
