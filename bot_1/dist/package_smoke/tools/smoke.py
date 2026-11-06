"""Exercise real persistent JSON-lines transport, reset and observation feed."""
import json
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
obs=json.loads((root/"examples"/"observation.json").read_text())
requests=[{"type":"reset","table_id":"demo"},obs,{"type":"observe","events":[{"id":1,"seat":0,"hand_id":1,"street":"preflop","action":"raise"}]},obs]
p=subprocess.run([sys.executable,str(root/"run_bot.py")],input="\n".join(json.dumps(r) for r in requests)+"\n",text=True,capture_output=True,cwd=root,timeout=10,check=True)
responses=[json.loads(x) for x in p.stdout.splitlines()]
assert len(responses)==4,(responses,p.stderr)
assert responses[0]==responses[2]=={"ok":True}
assert responses[1]["action"]=="raise"
assert 40<=responses[1]["amount"]<=2000
assert responses[3]["action"] in ("raise","call")
print(json.dumps({"success":True,"requests":4,"responses":responses},indent=2))
