"""Build the optional helper for this platform; never invoked by gameplay."""
import argparse
import pathlib
import platform
import subprocess
import sys

root=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument("--compiler",default="g++")
a=p.parse_args()
name="fast_equity.dll" if sys.platform=="win32" else "fast_equity.dylib" if sys.platform=="darwin" else "fast_equity.so"
flags=["-O3","-std=c++17","-shared"]
flags+= ["-static"] if sys.platform=="win32" else ["-fPIC"]
command=[a.compiler,*flags,str(root/"pokerbot"/"fast_equity.cpp"),"-o",str(root/"assets"/name)]
print("Building original helper for",platform.platform(),flush=True)
subprocess.run(command,cwd=root,check=True)
print(root/"assets"/name)
