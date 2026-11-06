"""Build our original optional native backend; no third-party poker code."""
from __future__ import annotations
import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build(trainer=False):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise SystemExit("A C++17 compiler (g++ or clang++) is required to build the optional native backend.")
    windows = sys.platform == "win32"
    target = ROOT / "native" / (("b13_train.exe" if windows else "b13_train") if trainer else ("bot13.dll" if windows else "libbot13.so"))
    args = [compiler, "-std=c++17", "-O3", "-DNDEBUG", "-Wall", "-Wextra", str(ROOT / "native" / "bot13.cpp"), "-o", str(target)]
    if trainer:
        args.append("-DB13_TRAINER")
    else:
        args += ["-shared"]
        if not windows:
            args += ["-fPIC", "-fvisibility=hidden"]
    if windows:
        args += ["-static-libgcc", "-static-libstdc++"]
    subprocess.run(args, check=True, cwd=ROOT)
    print(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", action="store_true")
    build(parser.parse_args().trainer)
