"""Record runtime imports, frozen hashes, and isolated startup cost."""
import ast
import json
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
runtime=['agent.py','cards.py','features.py','equity.py','icm.py','pushfold.py','main.py']
imports={}
for name in runtime:
    nodes=list(ast.walk(ast.parse((ROOT/name).read_text())))
    imports[name]=sorted({alias.name for n in nodes if isinstance(n,ast.Import) for alias in n.names}|
                         {n.module or '' for n in nodes if isinstance(n,ast.ImportFrom)})
frozen=json.loads((ROOT/'.control/frozen.json').read_text())
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in frozen.items())
code="import sys,time;sys.path.insert(0,'.');t=time.perf_counter();import agent;u=time.perf_counter();a=agent.Agent();print(u-t,time.perf_counter()-u)"
result=subprocess.run([sys.executable,'-I','-S','-c',code],cwd=ROOT,capture_output=True,text=True,check=True)
times=list(map(float,result.stdout.split()))
report={'imports':imports,'frozen_hashes_match':True,'import_seconds':times[0],
        'constructor_seconds':times[1], 'python':sys.version,
        'note':'One isolated-process local startup observation. Import/model initialization is outside the in-process decision benchmark. Actual host timing policy unknown.'}
(ROOT/'results/runtime_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
