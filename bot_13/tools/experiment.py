"""Run an explicit, bounded batch of independent LOCAL tuning/evaluation tasks.

No network or cluster operations. Use only inside our authorized compute job
for large batches. Each task writes its full reproducible seeded table results.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_task(task):
    from lab.benchmark import run_benchmark
    name = task["name"]
    args = dict(task["args"])
    args["progress_every"] = 0
    start = time.perf_counter()
    result = run_benchmark(**args)
    return name, result, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        p.error("bounded team resource limit: 1..8 CPU workers")
    data = json.loads(a.plan.read_text(encoding="utf-8"))
    tasks = data["tasks"]
    if len({t["name"] for t in tasks}) != len(tasks):
        p.error("task names must be unique")
    if any(not t["name"].replace("_", "").replace("-", "").isalnum() for t in tasks):
        p.error("task names must be simple local filenames")
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": hashlib.sha256(a.plan.read_bytes()).hexdigest(),
                "phase": data["phase"], "workers": a.workers,
                "completed": [], "errors": []}
    (a.output / "plan.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run_task, task): task["name"] for task in tasks}
        for f in as_completed(futures):
            try:
                name, result, elapsed = f.result()
                (a.output / (name + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
                manifest["completed"].append({"name": name, "seconds": elapsed, "summary": result["summary"]})
                print("completed", name, round(elapsed, 2), "seconds", flush=True)
            except Exception as e:
                name = futures[f]
                manifest["errors"].append({"name": name, "type": type(e).__name__, "message": str(e)})
                print("FAILED", name, type(e).__name__, str(e), flush=True)
            (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if manifest["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
