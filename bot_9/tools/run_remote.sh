#!/bin/bash
#SBATCH --job-name=bot_9-equity
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=bot_9-equity-%j.log
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_9"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data assets measurements
g++ -std=c++17 -O3 -DNDEBUG tools/generate_equity.cpp -o generate_equity
./generate_equity validate
for shard in $(seq 0 6); do
    ./generate_equity samples "data/train_${shard}.csv" 60000 256 "$((91000 + shard))" assets/preflop.csv > "data/train_${shard}.log" 2>&1 &
done
./generate_equity samples data/holdout.csv 18000 1024 719238 assets/preflop.csv > data/holdout.log 2>&1 &
wait
python3 tools/train_equity.py --data data/train_*.csv --holdout data/holdout.csv --out assets/equity_model.json.gz --report measurements/equity_model.json --trees 40 --depth 17 --leaf 10
echo 'BOT_9_EQUITY_DONE'
