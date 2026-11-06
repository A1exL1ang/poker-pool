#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --job-name=bot_14_equity
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:45:00
#SBATCH --output=bot_14_equity-%j.out
#SBATCH --error=bot_14_equity-%j.err
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_14"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p build assets measurements
python3 tools/environment.py
g++ -std=c++17 -O3 -march=native -fopenmp tools/equity_assets.cpp -o build/equity_assets
./build/equity_assets --trials 20000 --multiway-trials 20000 --seed 140014 --output assets/preflop.json
python3 tools/train_equity.py --prepare-ranks assets/preflop.json --ranks-output build/preflop_ranks.txt
g++ -std=c++17 -O3 -march=native -fopenmp tools/postflop_dataset.cpp -o build/postflop_dataset
./build/postflop_dataset --rows 300000 --trials 256 --seed 140120 --ranks build/preflop_ranks.txt --output build/postflop_train.bin
./build/postflop_dataset --rows 24000 --trials 2048 --seed 914007 --ranks build/preflop_ranks.txt --output build/postflop_validation.bin
python3 tools/train_equity.py --train build/postflop_train.bin --validation build/postflop_validation.bin --output assets/equity_model.json --report measurements/equity_validation.json
