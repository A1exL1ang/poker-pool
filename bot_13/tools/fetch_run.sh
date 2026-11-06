#!/usr/bin/env bash
# Fetch only an explicitly named run and job OWNED by bot_13.
set -euo pipefail
cd /mnt/c/Users/alexl/Code/poker-pool/.workers/bot_13
run=${1:?run name required}
job=${2:?owned job ID required}
[[ "$run" =~ ^[a-z0-9_]+$ && "$job" =~ ^[0-9]+$ ]] || exit 2
host=acliang@orcd-login.mit.edu
ssh_options=(-o BatchMode=yes -o ControlPath=/tmp/poker-pool-7kpPj9/control-%C -o UserKnownHostsFile=/mnt/c/Users/alexl/.ssh/known_hosts)
ssh "${ssh_options[@]}" "$host" "cd ~/poker-pool-extended-20261106/bot_13/runs/$run && if test -d audit; then tar -czf results.tgz results src/assets src/native/libbot13.so audit src/dist; else tar -czf results.tgz results src/assets src/native/libbot13.so; fi"
scp "${ssh_options[@]}" "$host:~/poker-pool-extended-20261106/bot_13/runs/$run/results.tgz" "measurements/remote_$run.tgz"
ssh "${ssh_options[@]}" "$host" "sacct -j $job -n -P --format=JobID,JobName,State,Elapsed,AllocCPUS,Submit,Start,End"
