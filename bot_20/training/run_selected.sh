#!/usr/bin/env bash
# Regenerate the selected bot_20 equity assets, never final evaluation labels.
# Run from bot_20's private workspace inside its OWN bounded compute allocation.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
if ! [[ "$OMP_NUM_THREADS" =~ ^[1-8]$ ]]; then
    echo "OMP_NUM_THREADS must be an integer from 1 through 8" >&2
    exit 2
fi
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p pokerbot/assets measurements training/data training/reproduced_selected
g++ -O3 -std=c++17 -fopenmp training/generate.cpp -o training/generate
training/generate --mode order --samples 30000 --seed 103117 --out training/data/order.csv
training/generate --mode preflop --order training/data/order.csv --samples 12000 --seed 327119 --out pokerbot/assets/preflop.json
training/generate --mode postflop --order training/data/order.csv --rows 1000000 --samples 512 --seed 471029 --out training/data/train_selected.npy
training/generate --mode postflop --order training/data/order.csv --rows 10000 --samples 1024 --seed 781237 --out training/data/validation.npy
python3 training/experiment_equity.py --train training/data/train_selected.npy --validation training/data/validation.npy --preflop pokerbot/assets/preflop.json --candidates street48 --output-dir training/reproduced_selected --report measurements/equity_selected_reproduction.json
cp training/reproduced_selected/street48.json.gz pokerbot/assets/equity_model.json.gz
