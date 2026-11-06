"""Replay observed LOCAL decisions for honest end-to-end wall-clock timing.

Read/init happens before timing. GC remains in its default state unless the
explicit diagnostic --disable-gc option is supplied. Replay does not represent
fresh independent poker tournaments and is not playing-strength evidence.
"""

import argparse
from collections import Counter, defaultdict
import gc
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot13.agent import Agent


def stats(values):
    values = sorted(values)
    def at(q):
        return values[min(len(values) - 1, int(q * len(values)))] / 1000
    return {"count": len(values), "median_us": at(0.5), "p95_us": at(0.95),
            "p99_us": at(0.99), "max_us": values[-1] / 1000,
            "mean_us": sum(values) / len(values) / 1000,
            "over_1ms": sum(v > 1_000_000 for v in values)} if values else {"count": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--config", default="{}")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--disable-gc", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    opener = gzip.open if a.corpus.suffix == ".gz" else open
    with opener(a.corpus, "rt", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]
    if a.limit:
        corpus = corpus[:a.limit]
    buckets = defaultdict(list)
    all_times, slowest = [], []
    metrics = Counter()
    kernel = None
    gc_start = gc.isenabled()
    if a.disable_gc:
        gc.disable()
    try:
        for rep in range(a.repeats):
            previous = object()
            for i, item in enumerate(corpus):
                table = item.get("table_id", 0)
                if table != previous:
                    if kernel is not None:
                        metrics.update({k: getattr(kernel, k) for k in ("decisions", "fallbacks", "native_errors", "native_calls")})
                    kernel = Agent(seed=int(item.get("seed", 13)) * 17, config=json.loads(a.config))
                    previous = table
                obs = item["observation"]
                start = perf_counter_ns()
                action = kernel.act(obs)
                elapsed = perf_counter_ns() - start
                all_times.append(elapsed)
                n = sum(not p["folded"] and p["seat"] != obs["hero"] and p["stack"] + p["committed"] > 0 for p in obs["players"])
                buckets[obs["street"] + "_" + str(n) + "opp"].append(elapsed)
                if elapsed > 1_000_000:
                    slowest.append({"rep": rep, "index": i, "elapsed_us": elapsed / 1000,
                                    "street": obs["street"], "opponents": n, "action": action})
            if kernel is not None:
                metrics.update({k: getattr(kernel, k) for k in ("decisions", "fallbacks", "native_errors", "native_calls")})
                kernel = None
    finally:
        if gc_start:
            gc.enable()
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "corpus": str(a.corpus), "corpus_sha256": hashlib.sha256(a.corpus.read_bytes()).hexdigest(),
              "repeats": a.repeats, "gc_disabled_diagnostic": a.disable_gc,
              "config": json.loads(a.config), "summary": stats(all_times),
              "agent_counters": dict(metrics),
              "cases": {k: stats(v) for k, v in sorted(buckets.items())},
              "overrun_samples": sorted(slowest, key=lambda x: -x["elapsed_us"])[:50]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "agent_counters": result["agent_counters"]}, indent=2))


if __name__ == "__main__":
    main()
