#!/bin/bash
#SBATCH --job-name=bot_16-equity-v1
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=bot_16-equity-%j.log
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_16"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=8
date -u
g++ -O3 -fopenmp training/generate.cpp -o training/generate
training/generate dataset training/train.csv 300000 384 161001 assets/strength.txt
date -u
training/generate dataset training/validation.csv 8000 2048 161002 assets/strength.txt
training/generate dataset training/test.csv 8000 2048 161003 assets/strength.txt
training/generate preflop training/preflop.csv 16000 30000 161004 assets/strength.txt
date -u
python3 training/train.py --train training/train.csv --validation training/validation.csv --preflop training/preflop.csv
date -u
tar -czf bot_16-equity-results.tgz assets/equity_model.json assets/preflop.json training/model_metrics.json training/validation.csv training/test.csv training/preflop.csv
