"""Evaluate a FROZEN runtime against a fresh, independently generated dataset.

This script deliberately calls actual estimate(), including exact rare-board
overrides. It does not fit, tune, select, or change a model. The head must freeze
the final candidate before generating/reading the independent final labels.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns, thread_time_ns

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from equity.model import estimate,MODEL_AVAILABLE,PREFLOP_AVAILABLE,PREMIUM_AVAILABLE,COMPILED_AVAILABLE,MODEL_STATUS
from equity.features import FEATURE_VERSION


def quantile(values,q):
    ordered=sorted(values)
    return ordered[min(len(ordered)-1,int(q*len(ordered)))]


def summarize(rows):
    errors=[r["prediction"]-r["equity"] for r in rows]
    absolute=[abs(e) for e in errors]
    return {"n":len(rows),"mae":statistics.mean(absolute),
            "rmse":statistics.mean(e*e for e in errors)**.5,
            "bias":statistics.mean(errors),"p95_absolute":quantile(absolute,.95),
            "p99_absolute":quantile(absolute,.99),"max_absolute":max(absolute)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--output",default="training/final_evaluation.json")
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--label-trials",type=int,default=8192)
    parser.add_argument("--expected-model-sha",required=True,
                        help="SHA256 fixed before inspecting this final dataset")
    parser.add_argument("--purpose",default="independent final evaluation of frozen actual runtime")
    args=parser.parse_args()
    model_path=ROOT/"equity"/"model.json"
    model_sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_sha!=args.expected_model_sha:
        raise SystemExit("Frozen model hash mismatch; no final evaluation was run")
    if not all((MODEL_AVAILABLE,PREFLOP_AVAILABLE,PREMIUM_AVAILABLE,COMPILED_AVAILABLE)):
        raise SystemExit("Missing/incompatible runtime asset; no final evaluation was run")
    with open(args.data,newline="") as handle:
        data=list(csv.DictReader(handle))
    # Warm code objects using fixed cards that are not read from final labels.
    for _ in range(100):estimate((48,45),(0,5,10),.25,2)
    results=[];wall=[];cpu=[]
    for row in data:
        hole=(int(row["h0"]),int(row["h1"]))
        board=tuple(int(row[f"b{i}"]) for i in range(5) if int(row[f"b{i}"])>=0)
        fraction=float(row["range"]);opponents=int(row["opponents"])
        start=perf_counter_ns();cpu_start=thread_time_ns()
        prediction=estimate(hole,board,fraction,opponents)
        cpu.append((thread_time_ns()-cpu_start)/1000);wall.append((perf_counter_ns()-start)/1000)
        if not 0<=prediction<=1:raise AssertionError("invalid equity prediction")
        results.append({"cards":[*hole,*board],"street":len(board),"range":fraction,
                        "opponents":opponents,"equity":float(row["equity"]),"prediction":prediction})
    report={"purpose":args.purpose,
            "model_sha256":model_sha,"feature_version":FEATURE_VERSION,
            "model_status":MODEL_STATUS,"compiled_available":COMPILED_AVAILABLE,
            "data_file":Path(args.data).name,"data_sha256":hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "seed":args.seed,"label_trials":args.label_trials,
            "generator_distribution":"uniform street and range; uniform opponents1..7",
            "target_max_heads_up_standard_error":.5/(args.label_trials**.5),
            "overall":summarize(results),"by_street":{},"by_range":{},"by_opponents":{},
            "latency_us":{"n":len(wall),"median":statistics.median(wall),"p95":quantile(wall,.95),
                          "p99":quantile(wall,.99),"p999":quantile(wall,.999),"max":max(wall),
                          "over_1ms":sum(t>1000 for t in wall)},
            "thread_cpu_us":{"median":statistics.median(cpu),"p99":quantile(cpu,.99),"max":max(cpu),
                             "over_1ms":sum(t>1000 for t in cpu)},
            "thread_cpu_timing_note":("Per-call CPU clock is visibly quantized; do not interpret CPU percentiles/overruns" if statistics.median(cpu)==0 and max(cpu)>=1000 else "No obvious coarse CPU-clock quantization detected"),
            "largest_errors":sorted(results,key=lambda r:abs(r["prediction"]-r["equity"]),reverse=True)[:20]}
    for key in ("street","range","opponents"):
        for value in sorted({r[key] for r in results}):
            report["by_"+key][str(value)]=summarize([r for r in results if r[key]==value])
    Path(args.output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("model_sha256","overall","by_street","by_opponents","latency_us","thread_cpu_us")},indent=2))


if __name__=="__main__":main()
