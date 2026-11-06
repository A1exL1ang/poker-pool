"""Build and smoke-test an allowlisted, offline, standard-library-only bundle."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FILES=['agent.py','cards.py','features.py','equity.py','icm.py','main.py',
       'pushfold.py','assets/preflop_equity.json','assets/equity_model.json','README.md']
if (ROOT/'assets/pushfold.json').exists(): FILES.append('assets/pushfold.json')


def main():
    target=ROOT/'dist';target.mkdir(exist_ok=True)
    bundle=target/'bot_10.zip'
    manifest={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in FILES:z.write(ROOT/name,name)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    assert bundle.stat().st_size<100_000_000
    assert sum((ROOT/name).stat().st_size for name in FILES)<100_000_000
    with tempfile.TemporaryDirectory(prefix='bot_10_offline_',dir=ROOT/'.control') as directory:
        with zipfile.ZipFile(bundle) as z:z.extractall(directory)
        captures=json.loads((ROOT/'results/development_capture.json').read_text())
        sample=next(o for row in captures['rows'] for o in row['observations'] if o['board'])
        (Path(directory)/'fixture.json').write_text(json.dumps(sample))
        # -I excludes user site packages and PYTHONPATH; insert extracted root.
        code="import sys,json,socket;socket.socket=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('network forbidden'));sys.path.insert(0,'.');from agent import Agent;from equity import TREES;assert TREES and 'icm' in sys.modules;a=Agent();o=json.load(open('fixture.json'));r=a.act(o);assert r['action'] in ('fold','check','call','raise') and a.failures==0;a.reset();assert a.act({'to_call':0})=={'action':'check'};print('offline import/postflop/reset/fallback passed')"
        subprocess.run([sys.executable,'-I','-S','-c',code],cwd=directory,check=True)
    report={'zip_bytes':bundle.stat().st_size,'uncompressed_bytes':sum((ROOT/name).stat().st_size for name in FILES),'files':manifest,'offline_isolated_smoke':'passed'}
    (ROOT/'results/package.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
