#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --job-name=bot_14_live_timing
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=bot_14_live_timing-%j.out
#SBATCH --error=bot_14_live_timing-%j.err
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_14"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 tools/live_latency.py --tables 1200 --workers 8 --seed 3014014 --output measurements/live_latency_linux.json
python3 tools/live_latency.py --tables 300 --workers 8 --seed 4014014 --config '{"native":false}' --output measurements/live_latency_portable_linux.json
