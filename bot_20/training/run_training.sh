#!/usr/bin/env bash
# Run only in bot_20's private workspace under its own bounded allocation.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p pokerbot/assets measurements training/data
g++ -O3 -std=c++17 -fopenmp training/generate.cpp -o training/generate
training/generate --mode order --samples 30000 --seed 103117 --out training/data/order.csv
training/generate --mode preflop --order training/data/order.csv --samples 6000 --seed 227119 --out pokerbot/assets/preflop.json
training/generate --mode postflop --order training/data/order.csv --rows 180000 --samples 192 --seed 371029 --out training/data/train.npy
training/generate --mode postflop --order training/data/order.csv --rows 10000 --samples 1024 --seed 781237 --out training/data/validation.npy
python3 training/fit_equity.py --train training/data/train.npy --validation training/data/validation.npy --preflop pokerbot/assets/preflop.json --iterations 180 --leaves 31 --depth 7 --output pokerbot/assets/equity_model.json.gz --report measurements/equity_validation.json
# Do not generate/read final evaluation until root freezes the candidate.
