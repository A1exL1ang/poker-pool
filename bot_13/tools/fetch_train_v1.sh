#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/alexl/Code/poker-pool/.workers/bot_13
host=acliang@orcd-login.mit.edu
ssh_options=(-o BatchMode=yes -o ControlPath=/tmp/poker-pool-7kpPj9/control-%C -o UserKnownHostsFile=/mnt/c/Users/alexl/.ssh/known_hosts)
ssh "${ssh_options[@]}" "$host" 'cd ~/poker-pool-extended-20261106/bot_13/runs/train_v1 && tar -czf results.tgz results src/assets/preflop.json src/native/libbot13.so'
scp "${ssh_options[@]}" "$host:~/poker-pool-extended-20261106/bot_13/runs/train_v1/results.tgz" measurements/remote_train_v1.tgz
scp "${ssh_options[@]}" "$host:~/poker-pool-extended-20261106/bot_13/.control/bot_13-train-v1-28072473.out" .control/job_28072473.log
ssh "${ssh_options[@]}" "$host" 'sacct -j 28072473 -n -P --format=JobID,JobName,State,Elapsed,AllocCPUS,Submit,Start,End'
