"""Consolidate final evidence with current source hashes; no invented results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parent.parent
    bench = root / "benchmarks"
    def read(name):
        path = bench / name
        return json.loads(path.read_text()) if path.exists() else None
    sources = [root / "agent.py", root / "assets/preflop.json"] + sorted((root / "bot6").glob("*.py"))
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    tests = (bench / "test_results.txt").read_text()
    match = re.search(r"Ran (\d+) tests in ([\d.]+)s", tests)
    fields = []
    for prefix in ("final", "linux"):
        for field in ("mixed", "tight", "calling", "maniac", "self"):
            filename = f"{prefix}_{field}.json"
            data = read(filename)
            if data:
                fields.append({"file": filename, **{key: data[key] for key in (
                    "field", "tables", "seed", "mean_final_chip_share", "standard_error_chip_share",
                    "outright_win_rate", "agent_decisions", "latency_us", "enforce_1ms_timeout")},
                    "timeout_fallbacks": data["events"].get("timeout_fallback", 0)})
    latency = []
    for name in ("final_latency_windows.json", "final_latency_typed_windows.json", "linux_latency.json"):
        data = read(name)
        if data:
            latency.append({"file": name, **{key: data[key] for key in ("platform", "python", "input", "iterations", "wall")}})
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "runtime_sha256": hashes,
        "tests": {"file": "test_results.txt", "count": int(match[1]) if match else None,
                  "seconds": float(match[2]) if match else None, "passed": tests.rstrip().endswith("OK")},
        "exhaustive_evaluator": read("evaluator.json"), "latency": latency, "fields": fields,
        "interpretation": "Original synthetic local fixtures, not official contest results. Blinds double every 25 hands for these tests; later official blind sizes and payouts are unknown. Win/final chip share uses a WTA diagnostic. Timing excludes startup and JSON transport. Internal deadlines cannot guarantee 1 ms under OS scheduling.",
        "remote_job": read("engaging_job.json"),
    }
    (bench / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"tests": result["tests"], "latency_runs": len(latency), "field_runs": len(fields),
                      "runtime_files_hashed": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()
