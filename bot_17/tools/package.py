"""Build a small offline candidate archive; never submits or publishes it."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FILES=['agent.py','equity.py','risk.py','run_bot.py','CONTRACT.md','README.md',
       'data/__init__.py','data/preflop.json','data/equity_model.json','data/equity_predictor.py','native/poker_equity.dll',
       'native/poker_equity.so','native/equity.cpp','tools/build_native.py',
       'measurements/REPORT.md','measurements/selection.json','measurements/validation_summary.json',
       'data/equity_model_heldout.json','data/equity_sidepots_heldout.json']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'dist'/'bot_17.zip')
    p.add_argument('--checkpoint',action='store_true',help='Allow missing final assets for development snapshot')
    args=p.parse_args()
    if not args.checkpoint:
        for f in ['agent.py','equity.py','risk.py','CONTRACT.md','README.md','data/preflop.json']:
            if not (ROOT/f).is_file():
                raise SystemExit('Required deliverable missing: '+f)
    files=[f for f in FILES if (ROOT/f).is_file()]
    manifest={'name':'bot_17','interface':'local CONTRACT.md; official adapter unavailable',
              'offline':True,'checkpoint':args.checkpoint,
              'sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT/f,f)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    size=args.output.stat().st_size
    if size>=100_000_000:
        raise SystemExit('Archive exceeds conservative decimal100MB limit')
    print(json.dumps({'archive':str(args.output),'bytes':size,'files':len(files),'checkpoint':args.checkpoint}))


if __name__=='__main__':
    main()
