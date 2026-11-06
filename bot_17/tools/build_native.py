"""Build only this bot's optional native accelerator; no package downloads."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', default=None)
    args = parser.parse_args()
    compiler = args.compiler or shutil.which('g++') or shutil.which('clang++')
    if compiler is None:
        raise SystemExit('No C++ compiler found. Pure Python fallback remains runnable.')
    extension = '.dll' if os.name == 'nt' else '.so'
    output = ROOT / 'native' / ('poker_equity' + extension)
    flags = ['-std=c++17', '-O3', '-DNDEBUG', '-shared']
    if os.name == 'nt':
        flags += ['-static', '-static-libgcc', '-static-libstdc++']
    else:
        flags += ['-fPIC']
    cmd = [compiler, *flags, str(ROOT / 'native' / 'equity.cpp'), '-o', str(output)]
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(output)

if __name__ == '__main__':
    main()
