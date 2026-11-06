"""Run the held-out local stress suite; records explicit synthetic conditions."""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.league import league
from pokerbot.native import LIB

p=argparse.ArgumentParser()
p.add_argument("--mixed",type=int,default=1000)
p.add_argument("--each",type=int,default=250)
p.add_argument("--selfplay",type=int,default=200)
p.add_argument("--seed",type=int,default=20261106)
p.add_argument("--output",default="benchmarks/fixture_suite.json")
a=p.parse_args()
results=[]
root=Path(__file__).resolve().parents[1]
paths=[root/"agent.py",*sorted((root/"pokerbot").glob("*.py")),root/"pokerbot"/"fast_equity.cpp",root/"assets"/"preflop.json"]
meta={"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"native_accelerator":LIB is not None,"source_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
for j,(name,count) in enumerate([("mixed",a.mixed)]+[(x,a.each) for x in ("station","nit","loose","maniac","rock","random")]+[("selfplay",a.selfplay)]):
    if count<=0:
        continue
    r=league(count,a.seed+100003*j,lineup="mixed" if name=="selfplay" else name,selfplay=name=="selfplay")
    results.append(r)
    print(name,"tables",count,"mean_points",round(r["mean_points"],3),"SE",round(r["standard_error"],3),"illegal",r["illegal_actions"],flush=True)
    Path(a.output).write_text(json.dumps({**meta,"results":results},indent=2)+"\n")
