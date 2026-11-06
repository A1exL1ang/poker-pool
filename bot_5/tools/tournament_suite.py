"""Run a fixed held-out suite; simple baselines are not contest opponents."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    configs = [("mixed", 1024, 91000), ("tight", 256, 92000), ("station", 256, 93000), ("pressure", 256, 94000), ("random", 256, 95000), ("self", 128, 96000)]
    summary = []
    for lineup, tables, seed in configs:
        output = f"evidence/final_{lineup}.json"
        subprocess.run([sys.executable, "simulator.py", "--lineup", lineup, "--tables", str(tables), "--seed", str(seed), "--output", output], cwd=ROOT, check=True)
        result = json.loads((ROOT / output).read_text())
        summary.append({"lineup": lineup, "tables": tables, "mean_points": result["mean_points"], "standard_error": result["standard_error_points"], "normal_95pct_interval": result["normal_95pct_interval"], "equal_share": result["equal_share_points"], "errors": len(result["policy_errors"])})
    (ROOT / "evidence" / "tournament_suite.json").write_text(json.dumps({"note": "Synthetic blinds double every 25 hands, synthetic payouts 0.5/0.3/0.2. Opponents are simple original baselines. These are not official contest results.", "results": summary}, indent=2) + "\n")


if __name__ == "__main__":
    main()
