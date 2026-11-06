"""Optional bounded bot_1-only Linux build through configured SSH/Slurm.

Uses a NEW one-CPU preemptible allocation, never existing user allocations.
No credentials are read or printed. Config contains connection routing only.
All remote reads/writes are confined to the explicitly authorized bot_1 path.
"""
import argparse
import base64
import datetime
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/".control"/"ENGAGING.json").read_text(encoding="utf-8-sig"))
assert cfg["remote_workspace"]=="~/poker-pool-20261106-054844/bot_1"
assert cfg["partition"]=="mit_preemptable"
ssh=[cfg["executable"],"-d",cfg["distribution"],"--","ssh","-o",f"ControlPath={cfg['control_path']}","-o",f"UserKnownHostsFile={cfg['known_hosts']}","-o","StrictHostKeyChecking=yes","-o","ControlMaster=no","-o","BatchMode=yes","-o","ConnectTimeout=10",cfg["host"]]
state_path=root/".control"/"remote_build.json"
remote='"$HOME/poker-pool-20261106-054844/bot_1/native_build"'

def run(script):
    # Pass bytes so Windows does not translate shell newlines into CRLF.
    p=subprocess.run(ssh+["bash -s"],input=script.encode(),capture_output=True,timeout=30,cwd=root)
    if p.returncode:
        raise RuntimeError((p.stderr.strip() or p.stdout.strip()).decode(errors="replace"))
    return p.stdout.decode()

p=argparse.ArgumentParser()
p.add_argument("action",choices=("start","status","fetch","cancel"))
a=p.parse_args()
if a.action=="start":
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,11,6,6,43,tzinfo=datetime.timezone.utc)
    source=base64.b64encode((root/"pokerbot"/"fast_equity.cpp").read_bytes()).decode()
    job='''#!/bin/bash
#SBATCH --job-name=bot_1_native
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --output=compile-%j.out
#SBATCH --error=compile-%j.err
set -eu
cd "$HOME/poker-pool-20261106-054844/bot_1/native_build"
g++ --version | head -n 1
g++ -O3 -std=c++11 -shared -fPIC -static-libgcc -static-libstdc++ fast_equity.cpp -o fast_equity.so
python3 -c 'import ctypes; lib=ctypes.CDLL("./fast_equity.so"); assert lib.bot_api_version()==3; print("bot_1 Linux API smoke passed")'
sha256sum fast_equity.so
'''
    job64=base64.b64encode(job.encode()).decode()
    script=f"set -eu\numask 077\nmkdir -p {remote}\ncd {remote}\nbase64 -d > fast_equity.cpp <<'BOT1_SOURCE'\n{source}\nBOT1_SOURCE\nbase64 -d > compile.sbatch <<'BOT1_JOB'\n{job64}\nBOT1_JOB\nsbatch --parsable compile.sbatch\n"
    output=run(script).strip()
    jobid=output.split(";")[0].splitlines()[-1]
    assert jobid.isdigit(),output
    state={"job_id":jobid,"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"remote_directory":"~/poker-pool-20261106-054844/bot_1/native_build","new_allocation":True,"partition":"mit_preemptable","cpus":1,"memory":"1G","time_limit":"00:03:00"}
    state_path.write_text(json.dumps(state,indent=2)+"\n")
    print(json.dumps(state,indent=2))
else:
    state=json.loads(state_path.read_text())
    jobid=state["job_id"]
    assert jobid.isdigit()
    if a.action=="status":
        output=run(f"set -eu\ncd {remote}\nsqueue -h -j {jobid} -o '%i %T %M %R'\nsacct -n -X -j {jobid} --format=JobID,State,ExitCode,Elapsed\nif [ -f compile-{jobid}.out ]; then cat compile-{jobid}.out; fi\nif [ -f compile-{jobid}.err ]; then cat compile-{jobid}.err; fi\n")
        (root/"benchmarks"/"engaging_build.json").write_text(json.dumps({**state,"last_checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"own_job_status_and_logs":output},indent=2)+"\n")
        print(output)
    elif a.action=="fetch":
        content=run(f"set -eu\ncd {remote}\nbase64 -w 0 fast_equity.so\n")
        binary=base64.b64decode(content,validate=True)
        assert binary[:4]==b"\x7fELF"
        (root/"assets"/"fast_equity.so").write_bytes(binary)
        print("Fetched bot_1 Linux helper:",len(binary),"bytes")
    else:
        print(run(f"scancel {jobid}\n"))
        print("Cancelled only own bot_1 job",jobid)
