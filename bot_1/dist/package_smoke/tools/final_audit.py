"""Verify final evidence fingerprints and summarize results without retesting."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((root/"benchmarks"/name).read_text())
def hashes_match(result):
    for name,expected in result["source_sha256"].items():
        path=(root/name.replace("\\","/")).resolve()
        assert path.is_relative_to(root.resolve()),name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name

validation={name:read(name+".json") for name in ("validation_native","validation_python","validation_linux_native","validation_linux_python")}
for r in validation.values():
    assert r["success"] and r["failures"]==r["errors"]==0
assert validation["validation_native"]["skipped"]==validation["validation_linux_native"]["skipped"]==0
latency={name:read(name+".json") for name in ("latency_native","latency_python","latency_linux_native")}
for r in latency.values():
    hashes_match(r)
fixtures=read("fixture_suite.json")
hashes_match(fixtures)
assert sum(r["tables"] for r in fixtures["results"])==3200
assert sum(r["illegal_actions"] for r in fixtures["results"])==0
exhaustive=read("exhaustive_evaluator.json")
assert exhaustive["passed"] and exhaustive["hands"]==2598960
remote=read("engaging_build.json")
assert remote["job_id"]=="28063667" and "COMPLETED" in remote["own_job_status_and_logs"]
assert hashlib.sha256((root/"assets"/"fast_equity.so").read_bytes()).hexdigest() in remote["own_job_status_and_logs"]
summary={"audited_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"evidence_consistent_with_current_sources":True,"validation":validation,"exhaustive_five_card_hands":exhaustive["hands"],"equity_sampling_same_model":read("equity_accuracy.json")["all"],"latency":{name:r["all"] for name,r in latency.items()},"hard_one_millisecond_guarantee":False,"timing_limitations":"Observed wall-clock violations on every measured path; import/startup/transport and official timing hardware unverified.","fixture_tables":sum(r["tables"] for r in fixtures["results"]),"fixture_hands":sum(r["hands"] for r in fixtures["results"]),"fixture_decisions":sum(r["decisions"] for r in fixtures["results"]),"fixture_illegal_actions":0,"fixture_scores":[{k:r[k] for k in ("lineup","tables","mean_points","standard_error","normal_95_interval")} for r in fixtures["results"]],"contest_compatibility":"Unverified: custom explicit interface; official payouts, later blinds and timing boundary not provided.","remote_job":{"id":remote["job_id"],"status":"COMPLETED","cpu":1,"memory":"1G","limit":"00:03:00","actual_elapsed":"00:00:01","existing_user_allocations_used":False,"running_task_jobs":0}}
(root/"benchmarks"/"final_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
