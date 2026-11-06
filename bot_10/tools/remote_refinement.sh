#!/bin/bash
#SBATCH --job-name=bot_10-refinement
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=bot_10-refinement-%j.log
set -eu
cd "$HOME/poker-pool-extended-20261106/bot_10"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cp equity_model_8.json assets/equity_model.json
.venv/bin/python tools/refine.py 1500
./generate_equity 8000 512 99010 heldout.csv
.venv/bin/python tools/evaluate_equity.py heldout.csv equity_model_8.json equity_model.json
.venv/bin/python tools/latency.py --rounds 20 --output results/latency_linux_8.json
