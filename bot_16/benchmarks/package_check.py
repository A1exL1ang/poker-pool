"""Validate a frozen runtime ZIP in an isolated, network-denied Python process.

  python benchmarks/package_check.py dist/bot_16.zip --output benchmarks/package_check.json

No package is produced here. Both archive and uncompressed bytes must be below
100,000,000. Extraction occurs only in this workspace's private .control path.
The socket block is a Python-level validation guard, not a kernel sandbox.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.latency import load_observations, random_observations, stress_observations

REQUIRED = ['agent.py', 'bot/__init__.py', 'bot/policy.py', 'bot/equity.py', 'bot/evaluator.py',
            'bot/features.py', 'bot/features_base.py', 'bot/exact.py', 'bot/icm.py', 'bot/_equity_compiled.py', 'assets/equity_model.json',
            'assets/preflop.json', 'assets/preflop_narrow.json']
LIMIT = 100_000_000

# All imports used to establish the guard are standard library. Agent import
# occurs only after sys.path pruning, import allowlisting and network denial.
PROBE = r'''
import sys, sysconfig, os, json, socket, importlib.abc, time
from pathlib import Path
root = Path(sys.argv[1]).resolve()
probe = Path(sys.argv[2]).resolve()
stdlib_paths = {Path(sysconfig.get_path(k)).resolve() for k in ('stdlib', 'platstdlib')}
stdlib_paths.add(Path(sys.base_prefix).resolve() / 'DLLs')
for p in tuple(stdlib_paths):
    stdlib_paths.add(p / 'lib-dynload')
for entry in sys.path:
    if entry and entry.lower().endswith('.zip'):
        p = Path(entry).resolve()
        if p.is_relative_to(Path(sys.base_prefix).resolve()):
            stdlib_paths.add(p)
sys.path[:] = [str(root)] + [str(p) for p in sorted(stdlib_paths, key=str) if p.exists()]
assert all('site-packages' not in p.lower() for p in sys.path)
allowed = set(sys.stdlib_module_names) | {'agent', 'bot'}
class OnlyAllowed(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in allowed:
            raise ImportError('Non-standard-library dependency denied: ' + fullname)
        return None
sys.meta_path.insert(0, OnlyAllowed())
network_attempts = []
def deny(*args, **kwargs):
    network_attempts.append('socket API')
    raise RuntimeError('Network access denied during offline validation')
def audit(event, args):
    if event.startswith('socket.'):
        network_attempts.append(event)
        raise RuntimeError('Socket audit event denied during offline validation')
socket.socket = deny
socket.create_connection = deny
socket.getaddrinfo = deny
sys.addaudithook(audit)
try:
    socket.create_connection(('offline.invalid', 443))
except RuntimeError:
    pass
else:
    raise AssertionError('Network guard self-test failed')
assert network_attempts
network_attempts.clear()
t0 = time.perf_counter_ns()
from agent import Agent
from bot import equity
import agent as entry
import_ns = time.perf_counter_ns() - t0
assert Path(entry.__file__).resolve().is_relative_to(root)
assert equity.MODEL_AVAILABLE, 'model fallback used'
assert equity.PREFLOP_AVAILABLE, 'preflop fallback used'
assert equity.COMPILED_AVAILABLE, 'compiled model unavailable or mismatched'
assert getattr(equity, '_NARROW', None) is not None, 'narrow preflop asset missing'
rows = json.loads(probe.read_text(encoding='utf-8'))
agent = Agent(seed=16, strict=True)
actions = {}
max_ns = 0
for index, obs in enumerate(rows):
    started = time.perf_counter_ns()
    result = agent.act(obs)
    max_ns = max(max_ns, time.perf_counter_ns() - started)
    assert isinstance(result, dict), result
    kind = result.get('action')
    assert kind in ('fold', 'check', 'call', 'raise'), result
    assert not (kind == 'check' and obs['to_call'] > 0), result
    assert not (kind == 'call' and obs['to_call'] == 0), result
    if kind == 'raise':
        amount = result.get('amount')
        hero = obs['players'][obs['hero']]
        assert type(amount) is int and obs['can_raise'], result
        assert hero['street_bet'] + obs['to_call'] < amount <= obs['max_raise_to'], result
        assert amount >= obs['min_raise_to'] or amount == obs['max_raise_to'], result
    actions[kind] = actions.get(kind, 0) + 1
assert getattr(agent, 'errors', 0) == 0
assert getattr(agent, 'fallbacks', 0) == 0
assert not network_attempts, network_attempts
# Ensure runtime modules were loaded from the isolated extraction and not the
# working source checkout or an installed third-party package.
for name, module in list(sys.modules.items()):
    if name == 'agent' or name == 'bot' or name.startswith('bot.'):
        origin = getattr(module, '__file__', None)
        if origin:
            assert Path(origin).resolve().is_relative_to(root), (name, origin)
    origin = getattr(module, '__file__', None)
    if origin:
        assert 'site-packages' not in str(origin).lower(), (name, origin)
print(json.dumps(dict(ok=True, decisions=len(rows), actions=actions,
                     errors=agent.errors, fallbacks=agent.fallbacks,
                     network_attempts=len(network_attempts), network_guard_self_test=True,
                     model_available=equity.MODEL_AVAILABLE,
                     preflop_available=equity.PREFLOP_AVAILABLE, narrow_available=True,
                     compiled_available=equity.COMPILED_AVAILABLE,
                     import_ms=import_ns / 1e6, max_act_us=max_ns / 1000,
                     isolated_flags=dict(isolated=sys.flags.isolated, no_site=sys.flags.no_site),
                     standard_library_only=True, python=sys.version)))
'''


def check(archive_path, observation_path=None, timeout=60):
    archive = Path(archive_path).resolve()
    if not archive.is_relative_to(ROOT):
        raise ValueError('Archive must be in this bot workspace')
    if not archive.is_file():
        raise FileNotFoundError(archive)
    compressed = archive.stat().st_size
    if compressed >= LIMIT:
        raise ValueError('Archive exceeds strict decimal100MB limit')
    rows = list(stress_observations().values())
    rows.extend(random_observations(hands=32, seed=161001)[:256])
    if observation_path:
        rows.extend(load_observations(observation_path)[:256])
    control = (ROOT / '.control').resolve()
    control.mkdir(exist_ok=True)
    if not control.is_relative_to(ROOT):
        raise ValueError('Unsafe extraction parent')
    with zipfile.ZipFile(archive) as package:
        infos = package.infolist()
        expanded = sum(i.file_size for i in infos)
        if expanded >= LIMIT:
            raise ValueError('Uncompressed package exceeds strict decimal100MB limit')
        names = set()
        folded_names = set()
        for info in infos:
            raw = info.filename.replace(chr(92), '/')
            path = PurePosixPath(raw)
            if path.is_absolute() or '..' in path.parts or ':' in raw or chr(0) in raw:
                raise ValueError('Unsafe archive path: ' + raw)
            if any(part.endswith(('.', ' ')) for part in path.parts):
                raise ValueError('Ambiguous Windows archive path: ' + raw)
            if path.parts and path.parts[0].casefold() in ('.control', '.git', '.aws', '.ssh'):
                raise ValueError('Private/control metadata included in runtime ZIP: ' + raw)
            if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                raise ValueError('Archive symlinks are not allowed')
            normalized = path.as_posix()
            if normalized.casefold() in folded_names:
                raise ValueError('Duplicate archive member: ' + normalized)
            names.add(normalized)
            folded_names.add(normalized.casefold())
        missing = set(REQUIRED) - names
        if missing:
            raise ValueError('Missing runtime files: ' + repr(sorted(missing)))
        with tempfile.TemporaryDirectory(prefix='package_check_', dir=control) as temporary:
            target = Path(temporary).resolve()
            # Explicit verification before recursive extraction and subsequent
            # TemporaryDirectory cleanup; no paths outside this task can enter.
            if not target.is_relative_to(control):
                raise ValueError('Extraction path escaped private control directory')
            package.extractall(target)
            fixture_path = target / '_offline_probe_observations.json'
            fixture_path.write_text(json.dumps(rows), encoding='utf-8')
            result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', PROBE,
                                     str(target), str(fixture_path)],
                                    cwd=target, capture_output=True, text=True,
                                    timeout=timeout, check=False)
            if result.returncode != 0:
                raise RuntimeError('Isolated package probe failed:\n' + result.stdout + result.stderr)
            probe = json.loads(result.stdout.strip())
    return dict(archive=str(archive.relative_to(ROOT)),
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                compressed_bytes=compressed, uncompressed_bytes=expanded,
                strict_decimal_size_limit_bytes=LIMIT, members=len(infos), required_files=REQUIRED,
                probe=probe,
                note='Python socket APIs/audit events denied; this validation is not a kernel network sandbox. Probe latency is diagnostic, not the final isolated latency distribution.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive')
    p.add_argument('--observations')
    p.add_argument('--timeout', type=int, default=60)
    p.add_argument('--output', default='benchmarks/package_check.json')
    args = p.parse_args()
    report = check(args.archive, args.observations, args.timeout)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
