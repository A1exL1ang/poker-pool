"""Create and smoke-test the offline runtime archive; never submit it."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = ['agent.py','equity.py','icm.py','run_bot.py','README.md',
         'assets/equity_predict.py','assets/equity_model.json','assets/preflop.json',
         'assets/preflop_narrow.json','assets/rank_scores.zlib','assets/__init__.py']


def main():
    dest=ROOT/'dist'
    dest.mkdir(exist_ok=True)
    archive=dest/'bot_12.zip'
    manifest={}
    for name in FILES:
        data=(ROOT/name).read_bytes()
        manifest[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(ROOT/name,name)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    if archive.stat().st_size >= 100_000_000:
        raise RuntimeError('package exceeds conservative100MB decimal ceiling')
    # Temp stays under the authorized workspace. -I excludes user site packages.
    with tempfile.TemporaryDirectory(dir=dest,prefix='offline-check-') as d:
        with zipfile.ZipFile(archive) as z:
            z.extractall(d)
        program="""
import json,socket,sys
sys.path.insert(0,sys.argv[1])
def denied(*a,**k): raise AssertionError('network access during offline test')
socket.socket=denied
from agent import Agent
from equity import analyze
bot=Agent()
o={'hole':[48,49],'board':[],'hero':3,'button':0,'street':'preflop',
 'stacks':[2000,1990,1980,2000,2000,2000,2000,2000],
 'contributions':[0,10,20,0,0,0,0,0],
 'street_contributions':[0,10,20,0,0,0,0,0],
 'in_hand':[True]*8,'all_in':[False]*8,'pot':30,'current_bet':20,
 'to_call':20,'min_raise_to':40,'max_raise_to':2000,
 'legal_actions':['fold','call','raise'],'small_blind':10,'big_blind':20,
 'hand_number':0,'history':[]}
assert bot.act(o)['action']=='raise'
assert 0<=analyze([48,49],[0,5,10])['equity_random']<=1
p=dict(o,street='flop',hole=[48,49],board=[0,5,10],pot=60,
 contributions=[0,0,20,20,20,0,0,0],street_contributions=[0]*8,
 in_hand=[False,False,True,True,True,False,False,False],current_bet=0,
 to_call=0,min_raise_to=20,legal_actions=['check','raise'],history=[])
assert bot.act(p)['action'] in p['legal_actions']
from icm import call_threshold
assert abs(call_threshold(dict(hero=0,stacks=[900,0],pot=1100,payouts=[.7,.3]),1,900,1100)-.45)<1e-12
assert analyze([48,49],[0,5,10,15,20])['equity_random']>=0
assert bot.fallbacks==0
assert not any(k.split('.')[0] in ('numpy','scipy','sklearn') for k in sys.modules)
print(json.dumps({'offline_import':True,'preflop_decision':True,'postflop_decision':True,'postflop_model':True,'river_asset':True,'supplied_payout_icm':True,'scientific_runtime_dependencies':False}))
"""
        completed=subprocess.run([sys.executable,'-I','-c',program,d],cwd=d,
                                 capture_output=True,text=True,timeout=30,check=True)
        check=json.loads(completed.stdout)
    report={'archive':'dist/bot_12.zip','archive_bytes':archive.stat().st_size,
            'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'uncompressed_bytes':sum(v['bytes'] for v in manifest.values()),
            'files':manifest,'checks':check,'python':sys.version}
    (dest/'package_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))


if __name__=='__main__':
    main()
