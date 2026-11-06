"""Create and verify an offline runtime archive from an explicit allowlist."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODULES = ("__init__", "strategy", "preflop", "model", "derived", "native", "evaluator", "equity", "icm", "interface")


def runtime_files():
    files = [ROOT / "agent.py"]
    files += [ROOT / "pokerbot" / (name + ".py") for name in MODULES]
    files += [ROOT / "assets" / name for name in ("preflop.json", "pushfold.json", "equity_model.json")]
    for name in ("libb14_equity.so", "b14_equity.dll"):
        path = ROOT / "pokerbot" / "_native" / name
        if path.exists():
            files.append(path)
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"missing or escaped runtime file: {path}")
    return files


def hashes():
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in runtime_files()}


VERIFY_CODE = r'''
import builtins,importlib,json,pathlib,socket,subprocess,sys,time
root=pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
from agent import Agent
from pokerbot.preflop import TABLES
from pokerbot.model import MODEL
from pokerbot.strategy import NATIVE
assert pathlib.Path(sys.modules['agent'].__file__).resolve().is_relative_to(root)
assert TABLES.loaded and MODEL.loaded and TABLES.pushfold
bot=Agent(seed=140014)
def observation(board=(),hole=(48,49),cost=0):
    players=[dict(stack=1800,invested=200,committed=0,folded=False,all_in=False,starting_stack=2000) for _ in range(8)]
    for i in range(2,8):players[i].update(folded=True,invested=0,stack=2000)
    if not board:
        players[0].update(invested=10,committed=10,stack=1990)
        players[1].update(invested=20,committed=20,stack=1980)
        cost=10
    else:
        players[1].update(invested=200+cost,committed=cost,stack=1800-cost)
    return dict(hero=0,button=0,small_blind_seat=0,big_blind_seat=1,hand_id=1,
       street={0:'preflop',3:'flop',4:'turn',5:'river'}[len(board)],hole_cards=hole,board=board,
       players=players,pot=sum(p['invested'] for p in players),current_bet=20 if not board else cost,
       to_call=cost,min_raise_to=40 if not board else max(20,2*cost),can_raise=True,
       small_blind=10,big_blind=20,history=[],payouts=(.5,.3,.2,0,0,0,0,0))
cases=[observation(),observation((0,9,18)),observation((0,9,18,27),cost=100),observation((0,9,18,27,36),cost=200),observation((51,47,43,39,35),hole=(1,2),cost=100)]
def forbidden(*a,**k):raise AssertionError('decision attempted I/O, process or network access')
builtins.open=forbidden
socket.socket=forbidden
subprocess.Popen=forbidden
times=[]
for o in cases:
    t=time.perf_counter_ns();a=bot.decide(o);times.append(time.perf_counter_ns()-t)
    assert a['action'] in ('fold','check','call','raise')
    if a['action']=='check':assert o['to_call']==0
    if a['action']=='raise':
        maximum=o['players'][0]['stack']+o['players'][0]['committed']
        assert o['current_bet']<a['amount']<=maximum
        assert a['amount']>=o['min_raise_to'] or a['amount']==maximum
assert bot.decide(cases[-1])['action'] in ('call','raise')
print(json.dumps(dict(cases=len(cases),native=NATIVE.available,maximum_smoke_wall_ns=max(times),diagnostics=bot.diagnostics)))
'''


def verify(archive, manifest):
    workspace = ROOT / ".control" / "package_checks"
    workspace.mkdir(parents=True, exist_ok=True)
    records = {}
    for portable in (False, True):
        with tempfile.TemporaryDirectory(prefix="bot_14_", dir=workspace) as temporary:
            target = Path(temporary).resolve()
            # TemporaryDirectory recursively cleans only this checked private
            # descendant; never a path outside the intended workspace.
            if not target.is_relative_to(workspace.resolve()):
                raise ValueError("temporary package path escaped workspace")
            with zipfile.ZipFile(archive) as z:
                for info in z.infolist():
                    relative = Path(info.filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("archive path traversal")
                    if portable and "_native" in relative.parts:
                        continue
                    data = z.read(info)
                    if info.filename != "MANIFEST.json":
                        assert hashlib.sha256(data).hexdigest() == manifest["files"][info.filename]["sha256"]
                    path = target / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
            result = subprocess.run([sys.executable, "-I", "-c", VERIFY_CODE, str(target)], cwd=target, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError(result.stderr + result.stdout)
            records["portable" if portable else "native"] = json.loads(result.stdout)
            if portable:
                assert records["portable"]["native"] is False
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="bot_14_runtime.zip")
    p.add_argument("--report", default="measurements/package_validation.json")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--lock", help="create a candidate source/asset lock before held-out evaluation")
    p.add_argument("--config", default="{}")
    args = p.parse_args()
    files = runtime_files()
    entries = {str(path.relative_to(ROOT)).replace("\\", "/"): dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size) for path in files}
    manifest = dict(format="bot_14_runtime_v1", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), entry="agent.Agent.decide", official_interface_compatibility=False, gameplay_dependencies="Python standard library; packaged native backend optional", files=entries, total_uncompressed_bytes=sum(x["bytes"] for x in entries.values()))
    if manifest["total_uncompressed_bytes"] >= 100_000_000:
        raise ValueError("runtime exceeds conservative 100 MB decimal limit")
    if args.lock:
        from pokerbot.strategy import DEFAULTS
        selected = json.loads(args.config)
        if selected != DEFAULTS:
            raise ValueError("selected config differs from packaged Agent defaults")
        lock = dict(phase="candidate locked before untouched evaluation", created_utc=manifest["created_utc"], config=selected, files={k:v["sha256"] for k,v in entries.items()})
        path = ROOT / args.lock
        if path.exists():
            raise ValueError("refuse to overwrite an existing candidate lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lock, indent=2))
    output = ROOT / args.output
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            z.write(path, str(path.relative_to(ROOT)).replace("\\", "/"))
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    if output.stat().st_size >= 100_000_000:
        raise ValueError("archive exceeds conservative 100 MB decimal limit")
    records = verify(output, manifest) if args.verify else {}
    report = dict(manifest=manifest, archive=output.name, archive_bytes=output.stat().st_size, archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), offline_import_and_decision_checks=records)
    (ROOT / "bot_14_manifest.json").write_text(json.dumps(manifest, indent=2))
    path = ROOT / args.report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
