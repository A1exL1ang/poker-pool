"""Build and offline-test a whitelist-only gameplay ZIP, without submitting it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='dist/bot_15.zip')
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    files = {name: name for name in ('agent.py', 'INTERFACE.md', 'MEASUREMENTS.md',
                                    'artifacts/equity_preflop.json',
                                    'artifacts/equity_model.json.gz', 'artifacts/policy.json')}
    files.update({str(path.relative_to(root)).replace('\\', '/'): str(path.relative_to(root)).replace('\\', '/') for path in (root / 'bot').glob('*.py')})
    files['README.md'] = 'PACKAGE_README.md'
    contents = {name: (root / source).read_bytes() for name, source in files.items()}
    manifest = {'local_interface_version': 1, 'official_compatibility_claimed': False,
                'runtime': 'Python>=3.10, standard library, offline',
                'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                          for name, data in sorted(contents.items())}}
    contents['MANIFEST.json'] = json.dumps(manifest, indent=2).encode()
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    assert output.stat().st_size < 100_000_000
    destination = root / '.control' / ('package_check_' + str(time.time_ns()))
    destination.mkdir(parents=True)
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert all('..' not in Path(name).parts and not Path(name).is_absolute() for name in z.namelist())
        z.extractall(destination)
    code = r'''
import json, sys, hashlib
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
def audit(event, args):
    if event.startswith('socket.'):
        raise RuntimeError('Network forbidden during offline package test')
sys.addaudithook(audit)
for name, info in json.loads((root/'MANIFEST.json').read_text())['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == info['sha256']
from agent import Agent
from bot.protocol import validate_observation
a=Agent()
o=dict(hero=0,button=0,hand_id=0,street=0,hole=[48,49],board=[],stacks=[1990,1980],in_hand=[True,True],street_bets=[10,20],contributions=[10,20],pot=30,big_blind=20,small_blind=10,to_call=10,min_raise_to=40,max_raise_to=2000,can_raise=True,position=1,history=[])
validate_observation(o)
assert a.act(o)['action']=='raise'
o.update(street=3,hole=[0,5],board=[32,36,40,44,48],stacks=[100,0],street_bets=[0,100],contributions=[100,200],pot=300,to_call=100,min_raise_to=200,max_raise_to=100,can_raise=False)
validate_observation(o)
assert a.act(o)==dict(action='call',amount=100)
assert a.errors==0
a.reset()
assert a.decisions==0 and a.errors==0
print('offline package verified: hashes, imports, assets, actions, reset; sockets forbidden')
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(destination)],
                            cwd=destination, check=True, text=True, capture_output=True)
    report = dict(path=str(output.relative_to(root)), bytes=output.stat().st_size,
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest(), files=len(contents),
                  offline_test=result.stdout.strip(), no_submission=True)
    (root / 'artifacts/package_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
