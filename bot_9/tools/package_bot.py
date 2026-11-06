"""Build and verify a local runtime ZIP; never upload or submit it.

python tools/package_bot.py --output dist/bot_9.zip
Only the explicit runtime whitelist is included. Verification launches Python
with -I -S in an isolated extracted directory with networking disabled.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('bot.py', 'model.py', 'features.py', 'evaluator.py', 'equity.py', 'play.py')
MAX_BYTES = 100_000_000


def runtime_files(root=ROOT):
    root = Path(root)
    asset = ('assets/equity_model.bin' if (root / 'assets/equity_model.bin').is_file()
             else 'assets/equity_model.json.gz')
    files = SOURCE_FILES + ('assets/preflop.csv', asset)
    for name in files:
        path = root / name
        if not path.is_file():
            raise ValueError(f'Missing runtime file: {name}')
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Runtime file escapes package directory: {name}')
    return files


def check_import_graph(root=ROOT):
    """Inspect every runtime source; imports must be local modules or stdlib.

    Dynamic importing is forbidden in this small audited runtime. This is an
    import-dependency check rather than a general security proof.
    """
    root = Path(root)
    allowed = set(sys.stdlib_module_names) | {Path(p).stem for p in SOURCE_FILES}
    graph = {}
    for name in SOURCE_FILES:
        tree = ast.parse((root / name).read_text(encoding='utf-8'), filename=name)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError(f'Unexpected relative import in {name}')
                imports.add((node.module or '').split('.')[0])
            elif isinstance(node, ast.Call):
                dynamic = (isinstance(node.func, ast.Name) and node.func.id == '__import__')
                dynamic |= (isinstance(node.func, ast.Attribute) and node.func.attr in ('import_module', 'exec_module'))
                if dynamic:
                    raise ValueError(f'Dynamic import needs manual audit: {name}')
        external = imports - allowed
        if external:
            raise ValueError(f'Non-stdlib dependencies in {name}: {sorted(external)}')
        graph[name] = sorted(imports)
    return graph


SMOKE = r'''
import hashlib, json, pathlib, socket, sys
root = pathlib.Path(sys.argv[1]).resolve()
manifest = json.loads((root / 'MANIFEST.json').read_text())
for name, entry in manifest['files'].items():
    data = (root / name).read_bytes()
    assert len(data) == entry['bytes'], name
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], name
assert set(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()) == set(manifest['files']) | {'MANIFEST.json'}
network_attempts = []
def denied(*args, **kwargs):
    network_attempts.append(True)
    raise RuntimeError('Network access forbidden in offline smoke test')
socket.socket = denied
socket.create_connection = denied
socket.socketpair = denied
socket.getaddrinfo = denied
socket.gethostbyname = denied
socket.gethostbyname_ex = denied
socket.gethostbyaddr = denied
sys.path.insert(0, str(root))
from bot import Agent
agent = Agent(seed=190)
assert bool(getattr(agent.model, 'trees', None)), 'Expected a loaded offline model, got fallback'
checks = 0
for street in range(4):
    board = [0,5,10,15,20][:0 if street == 0 else street+2]
    for call in (0,100):
        committed = [0]*8
        committed[2] = call
        contributions = [20]*8
        contributions[2] += call
        for can_raise in (False,True):
            o = dict(hero=3,hole=[48,49],board=board,stacks=[1980]*8,
                committed=committed,contributions=contributions,
                folded=[False]*8,all_in=[False]*8,button=0,street=street,
                pot=sum(contributions),to_call=call,min_raise_to=max(20,2*call),
                max_raise_to=1980,can_raise=can_raise,bb=20,sb=10,hand_no=checks,
                payouts=None,history=([dict(seat=2,street=street,action='raise',amount=call)] if call else []))
            a = agent.act(o)
            assert a['action'] in ('fold','check','call','raise'), a
            if a['action'] == 'check': assert not call, a
            if a['action'] == 'raise':
                assert can_raise and isinstance(a.get('amount'),int), a
                assert max(committed)<a['amount']<=o['max_raise_to'], a
                assert a['amount']>=o['min_raise_to'] or a['amount']==o['max_raise_to'], a
            checks += 1
assert agent.errors == 0, agent.errors
import io
from play import main as play_main
old_stdin,old_stdout=sys.stdin,sys.stdout
stream=io.StringIO()
try:
    sys.stdin=io.StringIO(json.dumps({'reset':True,'seed':91})+'\n'+json.dumps(o)+'\n')
    sys.stdout=stream
    play_main()
finally:
    sys.stdin,sys.stdout=old_stdin,old_stdout
lines=[json.loads(line) for line in stream.getvalue().splitlines()]
assert len(lines)==2 and lines[0]=={'reset':True}, lines
assert lines[1]['action'] in ('fold','check','call','raise'), lines
assert not network_attempts, 'Bot attempted network access'
for name,module in list(sys.modules.items()):
    path = getattr(module,'__file__',None)
    if path and ('site-packages' in path or 'dist-packages' in path):
        raise AssertionError('Third-party module imported: '+name)
print(json.dumps(dict(smoke_decisions=checks,agent_errors=agent.errors,
    network_attempts=len(network_attempts),python_flags='-I -S',
    model_loaded=True,third_party_modules=0,jsonl_lines=len(lines))))
'''


def verify_archive(path, *, timeout=90):
    """Hash-check and smoke test an archive without reading the source checkout."""
    path = Path(path).resolve()
    if path.stat().st_size >= MAX_BYTES:
        raise ValueError('ZIP must be smaller than 100,000,000 bytes')
    with tempfile.TemporaryDirectory(prefix='bot9-package-', dir=ROOT) as tmp:
        destination = Path(tmp)
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate ZIP entries')
            if any(name.startswith(('/', '\\')) or '..' in Path(name).parts or '\\' in name or ':' in name for name in names):
                raise ValueError('Unsafe ZIP entry')
            archive.extractall(destination)
        manifest = json.loads((destination / 'MANIFEST.json').read_text(encoding='utf-8'))
        expected = set(SOURCE_FILES) | {'assets/preflop.csv', manifest['model_asset'], 'MANIFEST.json'}
        if set(names) != expected:
            raise ValueError('Unexpected runtime package contents')
        check_import_graph(destination)
        result = subprocess.run([sys.executable, '-I', '-S', '-c', SMOKE, str(destination)],
                                cwd=destination, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'Isolated runtime smoke failed:\n{result.stderr}')
        return json.loads(result.stdout)


def build_package(output, *, root=ROOT, verify=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Package output must stay in this bot workspace')
    imports = check_import_graph(root)
    names = runtime_files(root)
    contents = {name: (root / name).read_bytes() for name in names}
    manifest = {'format': 1, 'entrypoint': 'bot.Agent',
                'interface': 'Independent local interface; official adapter unavailable',
                'jsonl_runner': 'play.py; parsing/printing/construction are outside Agent.act timing',
                'model_asset': names[-1],
                'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                          for name, data in contents.items()}, 'imports': imports}
    contents['MANIFEST.json'] = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode('utf-8')
    if sum(len(data) for data in contents.values()) >= MAX_BYTES:
        raise ValueError('Uncompressed runtime must stay below 100,000,000 bytes')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(contents):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, contents[name])
    if output.stat().st_size >= MAX_BYTES:
        raise ValueError('Compressed runtime must stay below 100,000,000 bytes')
    report = {'archive': output.relative_to(ROOT).as_posix(),
              'archive_bytes': output.stat().st_size,
              'archive_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'uncompressed_bytes': sum(map(len, contents.values())),
              'files': sorted(contents), 'model_asset': names[-1],
              'stdlib_only_static_import_graph': imports}
    if verify:
        report['isolated_smoke'] = verify_archive(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dist/bot_9.zip')
    parser.add_argument('--report', default='measurements/packaging.json')
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    report = build_package(output)
    report_path = (ROOT / args.report).resolve()
    if not report_path.is_relative_to(ROOT):
        raise ValueError('Report path must stay in this bot workspace')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
