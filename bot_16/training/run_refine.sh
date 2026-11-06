#!/bin/bash
#SBATCH --job-name=bot_16-equity-v3
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=bot_16-v3-%j.log
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_16"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=8
date -u
g++ -O3 -fopenmp training/generate_hard.cpp -o training/generate_hard
training/generate_hard training/hard.csv 100000 768 161011 assets/strength.txt
python3 training/train.py --extra-train training/hard.csv --features-module training.features_v3 --iterations 180 --depth 8 --leaves 64 --output assets/equity_v3_180.json --metrics training/v3_180_metrics.json
python3 training/train.py --extra-train training/hard.csv --features-module training.features_v3 --iterations 320 --depth 8 --leaves 96 --output assets/equity_v3_320.json --metrics training/v3_320_metrics.json
date -u
tar -czf bot_16-refine-results.tgz assets/equity_v3_180.json assets/equity_v3_320.json training/v3_180_metrics.json training/v3_320_metrics.json
