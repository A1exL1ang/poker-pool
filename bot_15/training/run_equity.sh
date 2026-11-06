#!/bin/bash
# Run only in a fresh bounded bot_15 allocation; source/data stay private.
#SBATCH --job-name=bot_15-equity
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/equity_job_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
g++ -O3 -std=c++17 training/native_equity.cpp -o training/native_equity
training/native_equity data artifacts/equity_train.bin 240000 15101 192 &
p1=$!
training/native_equity data artifacts/equity_valid.bin 12000 15102 1024 &
p2=$!
training/native_equity data artifacts/equity_test.bin 12000 15103 1024 &
p3=$!
wait "$p1"
wait "$p2"
wait "$p3"
python3 training/equity_train.py --train artifacts/equity_train.bin --valid artifacts/equity_valid.bin --test artifacts/equity_test.bin
