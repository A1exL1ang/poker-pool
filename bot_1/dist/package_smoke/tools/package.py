"""Create a private, reviewable source bundle; never submits or publishes."""
import datetime
import hashlib
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
paths=[root/x for x in ("README.md","INTERFACE.md","RULES.txt","agent.py","run_bot.py")]
for directory in ("pokerbot","assets","examples","tests","tools","benchmarks"):
    # Post-build smoke proofs hash the ZIP, so keep them outside the ZIP.
    paths.extend(p for p in (root/directory).iterdir() if p.is_file() and not (directory=="benchmarks" and p.name.startswith("package_smoke")) and p.suffix in (".py",".cpp",".h",".json",".txt",".dll",".so",".dylib"))
paths=sorted(set(paths))
manifest={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"description":"Private independent bot_1 source/assets/tests/evidence bundle; no competition submission or publication","files":{p.relative_to(root).as_posix():{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}}
dist=root/"dist"
dist.mkdir(exist_ok=True)
(dist/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
target=dist/"bot_1.zip"
with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in paths:
        z.write(p,p.relative_to(root).as_posix())
    z.write(dist/"MANIFEST.json","MANIFEST.json")
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
assert target.stat().st_size<100_000_000
print(json.dumps({"bundle":str(target),"bytes":target.stat().st_size,"files":len(paths)+1,"sha256":hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))
