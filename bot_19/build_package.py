"""Build a reproducible offline runtime archive; never pushes or submits."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    from equity_model import asset_status
    status=asset_status()
    if not status['postflop_model'] or not status['preflop_table'] or status.get('inference_backend') != 'generated':
        raise SystemExit('Refusing distribution without both trained assets: '+repr(status))
    names=['agent.py','icm.py','poker_eval.py','equity_model.py','equity_features.py','INTERFACE.md','README.md','RESULTS.md','SIMULATOR.md','RULES.txt',
           'assets/preflop.json','assets/equity_model.json']
    # Generated inference source and optional calibration assets are included
    # explicitly from this bot's assets directory, never from other workspaces.
    for path in sorted((ROOT/'assets').glob('*.py')):
        names.append(path.relative_to(ROOT).as_posix())
    for path in sorted((ROOT/'assets').glob('*.json')):
        name=path.relative_to(ROOT).as_posix()
        if name not in names:
            names.append(name)
    manifest={'bot':'bot_19','interface':'independent local contract; no official compatibility claimed',
              'runtime_dependencies':'Python standard library only','asset_status':status,
              'files':{n:{'bytes':(ROOT/n).stat().st_size,'sha256':hashlib.sha256((ROOT/n).read_bytes()).hexdigest()} for n in names}}
    dest=ROOT/'dist'/'bot_19.zip'
    dest.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in names:
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            archive.writestr(info,(ROOT/name).read_bytes())
        info=zipfile.ZipInfo('MANIFEST.json',date_time=(2026,1,1,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        archive.writestr(info,json.dumps(manifest,indent=2)+'\n')
    if dest.stat().st_size>=100000000 or sum(v['bytes'] for v in manifest['files'].values())>=100000000:
        raise SystemExit('package exceeds 100 MB')
    # Fresh extraction under this workspace; -I excludes working-dir and
    # user-site imports so accidental local or installed dependencies are caught.
    with tempfile.TemporaryDirectory(prefix='package_smoke_',dir=ROOT/'.control') as tmp:
        assert Path(tmp).resolve().is_relative_to((ROOT/'.control').resolve())
        with zipfile.ZipFile(dest) as archive:
            if any(Path(n).is_absolute() or '..' in Path(n).parts for n in archive.namelist()):
                raise SystemExit('unsafe archive member')
            archive.extractall(tmp)
        smoke=r'''
import builtins, json, pathlib, socket, sys
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
def denied(*a,**k): raise AssertionError('network unavailable')
socket.socket=denied
from agent import Agent
from equity_model import asset_status
assert asset_status()['postflop_model'] and asset_status()['preflop_table']
o={'hand_id':0,'street':'preflop','hero':0,'button':7,'hole':[48,49],
'board':[],'stacks':[2000,2000,2000,2000,2000,2000,1990,1980],
'bets':[0,0,0,0,0,0,10,20],'contributions':[0,0,0,0,0,0,10,20],
'folded':[False]*8,'pot':30,'to_call':20,'min_raise_to':40,'max_raise_to':2000,
'can_raise':True,'legal_actions':['fold','call','raise'],'small_blind':10,
'big_blind':20,'small_blind_seat':6,'big_blind_seat':7,'history':[],'payouts':None}
bot=Agent(19)
builtins.open=denied
pathlib.Path.open=denied
a=bot.act(o)
assert a['action']=='raise',a
assert 40<=a['amount']<=2000,a
assert bot.fallbacks==0
o.update(street='river',board=[0,5,10,15,20],pot=600,to_call=0,
bets=[0]*8,contributions=[75]*8,legal_actions=['check','raise'])
a=bot.act(o)
assert a['action'] in o['legal_actions'],a
assert bot.fallbacks==0
print(json.dumps({'offline_import_and_actions':'passed','asset_status':asset_status()}))
'''
        result=subprocess.run([sys.executable,'-I','-c',smoke,tmp],cwd=tmp,text=True,capture_output=True,check=True)
    report={'archive':'dist/bot_19.zip','bytes':dest.stat().st_size,
            'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'source_file_count':len(names),
            'uncompressed_bytes':sum(v['bytes'] for v in manifest['files'].values()),
            'validation':json.loads(result.stdout)}
    (ROOT/'measurements'/'package.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
