"""Compare deployment tree-prefix budgets using validation ONLY."""
import gc
import gzip
import json
from pathlib import Path
import statistics
import struct
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from bot.equity import features


def predict(x,trees):
    sums=[0.,0.,0.,0.]
    for node in trees:
        while node[0]>=0:node=node[2] if x[node[0]]<=node[1] else node[3]
        v=node[1]
        for k in range(4):sums[k]+=v[k]
    return [v/len(trees) for v in sums]


def pct(x,q):return sorted(x)[int(q*(len(x)-1))]


def main():
    # Load all source trees regardless of the final runtime TREE_LIMIT.
    source=Path('artifacts/equity_model_full24.json.gz')
    if not source.exists():source=Path('artifacts/equity_model.json.gz')
    with gzip.open(source,'rt') as f:raw=json.load(f)['trees']
    roots=[]
    for fs,ts,ls,rs,vs in raw:
        nodes=[None]*len(fs)
        for i in range(len(fs)-1,-1,-1):
            k=fs[i];nodes[i]=(k,ts[i],nodes[ls[i]],nodes[rs[i]]) if k>=0 else (-1,tuple(vs[-k-1]))
        roots.append(nodes[0])
    del raw,nodes
    with open('artifacts/equity_valid.bin','rb') as f:
        n,nf=struct.unpack('<ii',f.read(8));rows=list(struct.iter_unpack('<'+str(nf+4)+'f7i',f.read()))
    cases=[]
    for r in rows:
        h=list(r[nf+4:nf+6]);b=[c for c in r[nf+6:] if c>=0];cases.append((h,b,r[nf:nf+4]))
    # The source model is fixed. Prefix selection considers these validation
    # labels only; held-out predictions are not consulted in this script.
    reports=[]
    for limit in (8,12,16,24):
        trees=roots[:limit];errors=[];wall=[];cpu_start=time.process_time_ns()
        for h,b,y in cases:
            t=time.perf_counter_ns();p=predict(features(h,b),trees);wall.append((time.perf_counter_ns()-t)/1000)
            errors.extend(abs(a-b) for a,b in zip(p,y))
        cpu_avg=(time.process_time_ns()-cpu_start)/len(cases)/1000
        reports.append({'trees':limit,'validation_mae':statistics.mean(errors),'validation_p95_absolute_error':pct(errors,.95),'wall_us':{'median':statistics.median(wall),'p95':pct(wall,.95),'p99':pct(wall,.99),'over_1000us':sum(t>1000 for t in wall)},'process_cpu_average_us_including_error_aggregation':cpu_avg})
        print(reports[-1],flush=True)
    out={'selection_data':'validation only, seed15102','rows':len(rows),'representation':'nested immutable tuple tree nodes; fixed prefix of selected24-tree forest','candidates':reports}
    Path('artifacts/equity_runtime_selection.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
