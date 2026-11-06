"""Verify offline packaging in a fresh isolated Python process (run on compute).

Preserves its isolated extraction for inspection; never deletes unrelated paths.
Transport/startup times are recorded separately from warmed act() calls.
"""

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
from time import perf_counter
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent.parent

PROBE = r'''
import builtins,json,sys,time,socket
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
def forbidden(*a,**k): raise AssertionError("runtime external access")
socket.socket=forbidden
start=time.perf_counter_ns()
from agent import Agent
from bot13.interface import validate,legalize
agent=Agent(seed=130123)
initialization_ns=time.perf_counter_ns()-start
obs={"hand_id":0,"street":"preflop","hero":0,"button":0,
"small_blind":10,"big_blind":20,"small_blind_seat":1,"big_blind_seat":2,"seat_count":3,
"hole_cards":[51,47],"board":[],"pot":30,"to_call":20,"current_bet":20,
"min_raise_to":40,"max_raise_to":2000,"can_raise":True,"history":[],
"players":[{"seat":0,"stack":2000,"bet":0,"committed":0,"folded":False,"all_in":False},
{"seat":1,"stack":1990,"bet":10,"committed":10,"folded":False,"all_in":False},
{"seat":2,"stack":1980,"bet":20,"committed":20,"folded":False,"all_in":False}]}
river={"hand_id":1,"street":"river","hero":0,"button":0,"small_blind":10,"big_blind":20,
"small_blind_seat":0,"big_blind_seat":1,"seat_count":2,"hole_cards":[1,5],
"board":[32,36,40,44,48],"pot":500,"to_call":100,"current_bet":100,
"min_raise_to":200,"max_raise_to":1800,"can_raise":True,"history":[],
"players":[{"seat":0,"stack":1800,"bet":0,"committed":200,"folded":False,"all_in":False},
{"seat":1,"stack":1700,"bet":100,"committed":300,"folded":False,"all_in":False}]}
assert validate(obs) and validate(river)
python_agent=Agent(seed=130124,config={"native":False})
native_loaded=agent.native is not None
saved_open=builtins.open
builtins.open=forbidden
actions=[]
try:
    for policy,state in ((agent,obs),(agent,river),(python_agent,river)):
        start=time.perf_counter_ns()
        action=policy.act(state)
        elapsed=time.perf_counter_ns()-start
        assert action==legalize(action,state)
        assert policy.fallbacks==policy.native_errors==0
        actions.append({"action":action,"elapsed_ns":elapsed,"native":policy.native is not None})
    assert actions[1]["action"]=={"action":"call"}
    assert actions[2]["action"]=={"action":"call"}
    agent.reset()
    assert agent.decisions==0 and not agent.opponents and agent.seed==130123
finally:
    builtins.open=saved_open
assert native_loaded, "the intended native library failed to load in isolation"
print(json.dumps({"initialization_ns":initialization_ns,"native_loaded":native_loaded,
"actions":actions,"runtime_files_and_sockets_blocked":True}))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, default=ROOT / "dist" / "bot_13.zip")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    dest = ROOT / "measurements" / "isolated_packages" / uuid.uuid4().hex
    if not dest.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("extraction must remain inside the bot workspace")
    dest.mkdir(parents=True)
    with zipfile.ZipFile(a.package) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        names = set(archive.namelist())
        assert names == set(manifest["files"]) | {"MANIFEST.json"}
        for member in archive.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
                raise RuntimeError("unsafe package path")
            assert not any(x in member.filename for x in (".control", ".aws", ".ssh", "ENGAGING"))
        for name, expected in manifest["files"].items():
            value = archive.read(name)
            assert len(value) == expected["bytes"]
            assert hashlib.sha256(value).hexdigest() == expected["sha256"]
        archive.extractall(dest)
    started = perf_counter()
    run = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", PROBE, str(dest)],
                         cwd=dest, text=True, capture_output=True, check=True)
    probe = json.loads(run.stdout)
    result = {"package": str(a.package), "package_sha256": hashlib.sha256(a.package.read_bytes()).hexdigest(),
              "isolated_path": str(dest), "fresh_process_seconds": perf_counter() - started,
              "manifest_files_verified": len(manifest["files"]), "probe": probe,
              "transport_not_official": True}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
