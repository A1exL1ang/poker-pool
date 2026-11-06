#!/bin/bash
#SBATCH --job-name=bot_10-experiments
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=bot_10-experiments-%j.log
set -eu
cd "$HOME/poker-pool-extended-20261106/bot_10"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python train_equity.py --trees 8 --depth 16 --output equity_model_8.json
.venv/bin/python train_equity.py --trees 12 --depth 17 --output equity_model_12.json
.venv/bin/python tools/tune.py 600
.venv/bin/python tools/latency.py --rounds 10 --output results/latency_linux_24.json
