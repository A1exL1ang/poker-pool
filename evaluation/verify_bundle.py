"""Verify immutable source/model bytes before matches, inside the allocation."""
import hashlib
import importlib
import json
import platform
from pathlib import Path
from adapters import Adapter, Baseline, ROOT

def main():
    manifest=json.loads((ROOT/'evaluation/frozen_bots.json').read_text())
    missing=[];changed=[]
    for bot,files in manifest['bots'].items():
        for name,expected in files.items():
            p=ROOT/f'bot_{bot}'/name.replace('\\','/')
            if not p.exists():missing.append(str(p.relative_to(ROOT)))
            elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:changed.append(str(p.relative_to(ROOT)))
    baseline=json.loads((ROOT/'evaluation/baseline_manifest.json').read_text())
    base=ROOT/'evaluation/baseline_repo'/baseline['artifact']
    for name,expected in baseline['files'].items():
        p=base/name.replace('\\','/')
        if not p.exists():missing.append('baseline/'+name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:changed.append('baseline/'+name)
    one=Adapter(1)
    native=importlib.import_module(one.alias+'.native')
    report=dict(platform=platform.platform(),python=platform.python_version(),missing=missing,changed=changed,bot1_native_loaded=native.LIB is not None,baseline_commit=baseline['commit'])
    print(json.dumps(report))
    if missing or changed:raise SystemExit(1)

if __name__=='__main__':main()
