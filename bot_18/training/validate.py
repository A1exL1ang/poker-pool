"""Independent evaluator checks, trained model spot checks, and live latency.

python training/validate.py --native training/generate.exe --samples 20000
Results are local experiments; CPU scheduling overruns are reported explicitly.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from time import perf_counter_ns, thread_time_ns

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from equity.evaluator import evaluate,evaluate_reference
from equity.model import estimate,MODEL_AVAILABLE,PREFLOP_AVAILABLE,COMPILED_AVAILABLE,MODEL_STATUS


def quantile(values,q):
    return sorted(values)[min(len(values)-1,int(q*len(values)))]


def exact_river(hole,board):
    available=[c for c in range(52) if c not in hole and c not in board]
    hero=evaluate((*hole,*board));wins=0;total=0
    for i,a in enumerate(available):
        for b in available[i+1:]:
            value=evaluate((*board,a,b));wins+=(hero>value)+.5*(hero==value);total+=1
    return wins/total


def main():
    p=argparse.ArgumentParser();p.add_argument("--native");p.add_argument("--samples",type=int,default=20000);p.add_argument("--output",default="training/validation.json")
    args=p.parse_args();rng=random.Random(189471)
    hands=[rng.sample(range(52),5+i%3) for i in range(12000)]
    native_report={}
    if args.native:
        exe=str(Path(args.native).resolve())
        raw="\n".join(" ".join(map(str,h)) for h in hands)+"\n"
        proc=subprocess.run([exe,"--mode","check"],input=raw,text=True,capture_output=True,check=True)
        native=list(map(int,proc.stdout.split()))
        assert len(native)==len(hands)
        assert native==[evaluate(h) for h in hands]
        histogram=json.loads(subprocess.check_output([exe,"--mode","histogram"],text=True))
        expected=[1302540,1098240,123552,54912,10200,5108,3744,624,40]
        assert histogram==expected,(histogram,expected)
        native_report={"random_comparisons":len(hands),"five_card_histogram":histogram,"five_card_total":sum(histogram)}
    for h in hands[:3000]:assert evaluate(h)==evaluate_reference(h)
    states=[]
    for i in range(args.samples):
        hand=rng.sample(range(52),5+i%3);states.append((hand[:2],hand[2:],(.06,.12,.25,.45,1)[i%5],1+i%7))
    times=[];cpu_times=[]
    for state in states:
        start=perf_counter_ns();cpu_start=thread_time_ns();value=estimate(*state);cpu_times.append((thread_time_ns()-cpu_start)/1000);times.append((perf_counter_ns()-start)/1000)
        assert 0<=value<=1
    river=[]
    for _ in range(200):
        hand=rng.sample(range(52),7)
        target=exact_river(hand[:2],hand[2:]);prediction=estimate(hand[:2],hand[2:],1,1)
        river.append({"cards":hand,"exact":target,"prediction":prediction,"error":prediction-target})
    errors=[abs(r["error"]) for r in river]
    report={"model_available":MODEL_AVAILABLE,"preflop_available":PREFLOP_AVAILABLE,
            "compiled_available":COMPILED_AVAILABLE,"model_status":MODEL_STATUS,
            "independent_reference_comparisons":3000,"native":native_report,
            "latency_us":{"n":len(times),"median":statistics.median(times),"p95":quantile(times,.95),
                          "p99":quantile(times,.99),"p999":quantile(times,.999),"max":max(times),
                          "over_1ms":sum(t>1000 for t in times)},
            "thread_cpu_us":{"median":statistics.median(cpu_times),"p99":quantile(cpu_times,.99),"max":max(cpu_times),"over_1ms":sum(t>1000 for t in cpu_times)},
            "thread_cpu_timing_note":("Per-call CPU clock is visibly quantized; do not interpret CPU percentiles/overruns" if statistics.median(cpu_times)==0 and max(cpu_times)>=1000 else "No obvious coarse CPU-clock quantization detected"),
            "exact_random_range_river":{"n":len(river),"mae":statistics.mean(errors),
                                        "rmse":(statistics.mean(r["error"]**2 for r in river))**.5,
                                        "worst":sorted(river,key=lambda r:abs(r["error"]),reverse=True)[:10]}}
    Path(args.output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
