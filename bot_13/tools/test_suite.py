"""Run the complete rule/algorithm suite remotely and save machine-readable evidence."""

import argparse
import json
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {"platform": platform.platform(), "python": platform.python_version(),
               "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": [(str(t), reason) for t, reason in result.skipped],
               "elapsed_seconds": time.perf_counter() - start, "successful": result.wasSuccessful()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
