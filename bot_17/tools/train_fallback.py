"""Bounded local offline fallback training, one worker, no dependencies/downloads."""
import argparse
from pathlib import Path
import shutil
import subprocess
import os
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-rows',type=int,default=100000);p.add_argument('--validation-rows',type=int,default=6000)
    p.add_argument('--trees',type=int,default=80);p.add_argument('--train-samples',type=int,default=256)
    p.add_argument('--validation-samples',type=int,default=2048)
    p.add_argument('--depth',type=int,default=5)
    p.add_argument('--output',type=Path,default=ROOT/'data'/'equity_model.json')
    p.add_argument('--validation-output',type=Path,default=ROOT/'data'/'equity_validation.jsonl')
    args=p.parse_args();compiler=shutil.which('g++') or shutil.which('clang++')
    if not compiler:raise SystemExit('C++ compiler required for offline model generation')
    exe=ROOT/'native'/('train_fallback.exe' if os.name=='nt' else 'train_fallback')
    flags=['-O3','-std=c++17','-DNDEBUG']
    if os.name=='nt':flags+=['-static-libgcc','-static-libstdc++']
    subprocess.run([compiler,*flags,str(ROOT/'native'/'train_fallback.cpp'),'-o',str(exe)],cwd=ROOT,check=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.validation_output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.training.json')
    subprocess.run([str(exe),str(temporary),str(args.validation_output),str(args.train_rows),str(args.validation_rows),str(args.trees),str(args.train_samples),str(args.validation_samples),str(args.depth)],cwd=ROOT,check=True,timeout=1800)
    temporary.replace(args.output)
    if args.output.name=='equity_model.json':
        from compile_fallback import compile_model
        compile_model(args.output,ROOT/'data'/'equity_predictor.py')
    print(args.output)

if __name__=='__main__':main()
