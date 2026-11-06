#!/usr/bin/env bash
# Authorized transport only. Never attaches to or alters any existing jobs/master.
set -euo pipefail
cd /mnt/c/Users/alexl/Code/poker-pool/.workers/bot_13
host=acliang@orcd-login.mit.edu
base='poker-pool-extended-20261106/bot_13'
ssh_options=(-o BatchMode=yes -o ControlPath=/tmp/poker-pool-7kpPj9/control-%C -o UserKnownHostsFile=/mnt/c/Users/alexl/.ssh/known_hosts)
ssh "${ssh_options[@]}" "$host" "mkdir -p ~/$base/runs/train_v1/src ~/$base/.control"
scp "${ssh_options[@]}" checkpoints/initial_v1.zip "$host:~/$base/runs/train_v1/source.zip"
ssh "${ssh_options[@]}" "$host" 'cd ~/poker-pool-extended-20261106/bot_13/runs/train_v1/src && python3 -m zipfile -e ../source.zip . && sbatch --parsable --chdir="$HOME/poker-pool-extended-20261106/bot_13" --output="$HOME/poker-pool-extended-20261106/bot_13/.control/bot_13-train-v1-%j.out" tools/train_job.sh'
