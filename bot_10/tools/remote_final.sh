#!/bin/bash
#SBATCH --job-name=bot_10-final-eval
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=bot_10-final-eval-%j.log
set -eu
cd "$HOME/poker-pool-extended-20261106/bot_10"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python tools/final_evaluation.py
.venv/bin/python benchmark.py --tables 100 --seed 960000 --lineup challenge --capture --output results/final_capture.json
.venv/bin/python tools/latency.py --input results/final_capture.json --rounds 20 --output results/final_latency_linux.json
.venv/bin/python benchmark.py --tables 1000 --seed 970000 --lineup self --enforce-time --output results/final_timeout_enforced.json
