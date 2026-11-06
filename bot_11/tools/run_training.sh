#!/bin/bash
# Submit only as a NEW bot_11 private job. No existing allocation is reused.
# sbatch --partition=mit_preemptable --job-name=bot_11_equity --cpus-per-task=8 \
#   --mem=32G --time=02:00:00 --chdir=/home/acliang/poker-pool-extended-20261106/bot_11 \
#   --output=training-%j.log tools/run_training.sh
set -euo pipefail
cd /home/acliang/poker-pool-extended-20261106/bot_11
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p assets benchmarks training
g++ -O3 -std=c++17 -fopenmp tools/generate_equity.cpp -o training/generate_equity
training/generate_equity --mode postflop --states 600000 --rollouts 2048 --seed 1101201 --threads 8 --out training/train.bin
training/generate_equity --mode postflop --states 30000 --rollouts 8192 --seed 1102202 --threads 8 --out training/holdout.bin
training/generate_equity --mode preflop --rollouts 400000 --seed 1103303 --threads 8 --out assets/preflop.json
python3 -m tools.train_equity --train training/train.bin --test training/holdout.bin --trees 64 --leaf 6 --depth 23 --jobs 8
python3 -c 'from pokerbot.equity import EquityModel; from pokerbot.features import features; m=EquityModel(); print("export_reload", m.predict(features((48,49),(0,5,10))))'
