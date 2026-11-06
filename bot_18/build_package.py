"""Create and verify an offline-only runtime ZIP; never submits anything."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    files = [ROOT / 'agent.py', ROOT / 'run_local.py', ROOT / 'README.md']
    for directory in ('equity', 'policy'):
        files += [p for p in sorted((ROOT / directory).glob('*.py'))
                  if not p.stem.startswith(('compiled_v', 'model_v'))]
    files += [ROOT / 'equity' / name for name in ('model.json', 'preflop.json', 'premium.json')]
    for p in files:
        if not p.is_file():
            raise RuntimeError(f'Missing required file: {p.relative_to(ROOT)}')
    manifest = {str(p.relative_to(ROOT)).replace('\\', '/'): {
        'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    target = ROOT / 'dist' / 'bot_18_local.zip'
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT))
        z.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
    unpacked = sum(p.stat().st_size for p in files)
    if max(unpacked, target.stat().st_size) >= 100_000_000:
        raise RuntimeError('Package exceeds conservative 100MB bound')
    with tempfile.TemporaryDirectory(prefix='bot_18_package_', dir=target.parent) as tmp:
        with zipfile.ZipFile(target) as z:
            z.extractall(tmp)
        from sim.engine import Hand
        h = Hand([2000] * 8, 0)
        samples = [h.observation()]
        while h.street != 'flop':
            o = h.observation()
            h.apply({'action':'call' if o['to_call'] else 'check'})
        samples.append(h.observation())
        probe = ("from agent import Agent; from equity.model import MODEL_AVAILABLE,PREFLOP_AVAILABLE,PREMIUM_AVAILABLE,COMPILED_AVAILABLE; "
                 "assert all((MODEL_AVAILABLE,PREFLOP_AVAILABLE,PREMIUM_AVAILABLE,COMPILED_AVAILABLE)); "
                 "a=Agent(seed=18); observations=" + repr(samples) + "; "
                 "actions=[a.act(o) for o in observations]; assert a.errors==0; "
                 "assert all(x['action'] in ('check','call','raise','fold') for x in actions); "
                 "print('isolated stdlib import, compiled assets and live preflop/flop decisions: OK')")
        subprocess.run([sys.executable, '-S', '-c', probe], cwd=tmp, check=True)
    report = {'zip':str(target.relative_to(ROOT)), 'zip_bytes':target.stat().st_size,
              'unpacked_bytes':unpacked,'files':manifest,'offline_stdlib_import':'passed'}
    (ROOT / 'measurements').mkdir(exist_ok=True)
    (ROOT / 'measurements' / 'package.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))


if __name__ == '__main__':
    main()
