"""Persist full unittest output and a machine-readable summary."""
import argparse
import datetime
import io
import json
import sys
import time
import unittest
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from pokerbot.native import LIB

p=argparse.ArgumentParser()
p.add_argument("--output",default="benchmarks/validation_native")
a=p.parse_args()
stream=io.StringIO()
start=time.perf_counter()
suite=unittest.defaultTestLoader.discover(str(root/"tests"))
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
log=stream.getvalue()
Path(a.output+".txt").write_text(log,encoding="utf-8")
summary={"measured_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"native_accelerator":LIB is not None,"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"elapsed_seconds":time.perf_counter()-start,"success":result.wasSuccessful()}
Path(a.output+".json").write_text(json.dumps(summary,indent=2)+"\n")
print(log)
print(json.dumps(summary,indent=2))
sys.exit(0 if result.wasSuccessful() else 1)
