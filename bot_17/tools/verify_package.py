"""Verify archive hashes and isolated offline imports from an extracted package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,default=ROOT/'dist'/'bot_17.zip')
    p.add_argument('--output',type=Path,default=ROOT/'measurements'/'package_verification.json')
    args=p.parse_args()
    dest=Path(tempfile.mkdtemp(prefix='package_',dir=ROOT/'.control')).resolve()
    if not dest.is_relative_to(ROOT.resolve()):
        raise SystemExit('Extraction directory escaped workspace')
    with zipfile.ZipFile(args.archive) as z:
        for member in z.namelist():
            if not (dest/member).resolve().is_relative_to(dest):
                raise SystemExit('Unsafe archive member')
        z.extractall(dest)
    manifest=json.loads((dest/'MANIFEST.json').read_text())
    for f,expected in manifest['sha256'].items():
        assert hashlib.sha256((dest/f).read_bytes()).hexdigest()==expected,f
    script='''import sys,os,json
sys.path.insert(0,os.getcwd())
from agent import Agent
import equity
import builtins,io,socket
def forbid(*args,**kwargs): raise AssertionError('Unexpected gameplay I/O')
bot=Agent(17)
players=[dict(seat=0,stack=1980,bet=0,committed=20,folded=False,all_in=False),dict(seat=1,stack=1960,bet=20,committed=40,folded=False,all_in=False)]
o=dict(hand_id=1,hero=0,button=0,street=1,hole=[48,49],board=[0,5,10],players=players,pot=60,big_blind=20,small_blind=10,to_call=20,min_raise_to=40,max_raise_to=1980,can_raise=True,legal_actions=['fold','call','raise'],history=[])
builtins.open=forbid;io.open=forbid;socket.socket=forbid
action=bot.act(o)
assert action['action'] in o['legal_actions'] and bot.fallbacks==0
print(json.dumps(dict(backend=equity.BACKEND,action=action,fallbacks=bot.fallbacks)))
'''
    results={}
    for mode in ['native_if_available','pure_python']:
        env=os.environ.copy()
        if mode=='pure_python':env['POKER_PURE_PYTHON']='1'
        else:env.pop('POKER_PURE_PYTHON',None)
        run=subprocess.run([sys.executable,'-I','-c',script],cwd=dest,env=env,
                           capture_output=True,text=True,check=True,timeout=30)
        results[mode]=json.loads(run.stdout)
    report=dict(archive=str(args.archive),bytes=args.archive.stat().st_size,
                sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                verified_files=len(manifest['sha256']),extracted_to=str(dest),results=results,
                note='Isolated Python process imports only extracted runtime. Python file/network APIs disabled after initialization for decision. No publication/submission.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
