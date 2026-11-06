#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/alexl/Code/poker-pool/.workers/bot_13
host=acliang@orcd-login.mit.edu
ssh_options=(-o BatchMode=yes -o ControlPath=/tmp/poker-pool-7kpPj9/control-%C -o UserKnownHostsFile=/mnt/c/Users/alexl/.ssh/known_hosts)
ssh "${ssh_options[@]}" "$host" 'mkdir -p ~/poker-pool-extended-20261106/bot_13/runs/heldout_v4/src'
scp "${ssh_options[@]}" checkpoints/heldout_v4.zip "$host:~/poker-pool-extended-20261106/bot_13/runs/heldout_v4/source.zip"
ssh "${ssh_options[@]}" "$host" 'cd ~/poker-pool-extended-20261106/bot_13/runs/heldout_v4/src && python3 -m zipfile -e ../source.zip . && sbatch --parsable --chdir="$HOME/poker-pool-extended-20261106/bot_13" --output="$HOME/poker-pool-extended-20261106/bot_13/.control/bot_13-heldout-v4-%j.out" tools/heldout_job.sh'
