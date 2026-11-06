"""Verify/extract the private bundle inside bot_1 and run its own harness."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
bundle=root/"dist"/"bot_1.zip"
target=(root/"dist"/"package_smoke").resolve()
assert target.is_relative_to(root.resolve())
target.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    for entry in z.infolist():
        assert (target/entry.filename).resolve().is_relative_to(target)
    z.extractall(target)
manifest=json.loads((target/"MANIFEST.json").read_text())
for required in ("README.md","INTERFACE.md","agent.py","run_bot.py","pokerbot/agent.py","assets/preflop.json","assets/fast_equity.dll","assets/fast_equity.so","tools/package_smoke.py","tools/final_audit.py"):
    assert required in manifest["files"],required
for name,info in manifest["files"].items():
    path=target/name
    assert path.stat().st_size==info["bytes"],name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==info["sha256"],name
process=subprocess.run([sys.executable,str(target/"tools"/"smoke.py")],cwd=target,text=True,capture_output=True,timeout=15,check=True)
transport=json.loads(process.stdout)
assert transport["success"]
native=subprocess.run([sys.executable,"-c","from pokerbot.native import LIB; assert LIB is not None; assert LIB.bot_api_version()==3; print('native API 3 loaded')"],cwd=target,text=True,capture_output=True,timeout=10,check=True)
result={"success":True,"platform":sys.platform,"bundle_bytes":bundle.stat().st_size,"bundle_sha256":hashlib.sha256(bundle.read_bytes()).hexdigest(),"verified_manifest_files":len(manifest["files"]),"extracted_within_workspace":True,"transport":transport,"native":native.stdout.strip()}
(root/"benchmarks"/("package_smoke_"+sys.platform+".json")).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
