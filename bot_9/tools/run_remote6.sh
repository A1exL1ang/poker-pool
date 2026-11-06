#!/bin/bash
#SBATCH --job-name=bot_9-equity6
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=bot_9-equity6-%j.log
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_9"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data6 assets measurements
g++ -std=c++17 -O3 -DNDEBUG tools/generate_equity.cpp -o generate_equity
./generate_equity validate-made
for shard in $(seq 0 6); do
    ./generate_equity samples6 "data6/train_${shard}.csv" 60000 384 "$((193000 + shard))" assets/preflop.csv > "data6/train_${shard}.log" 2>&1 &
done
./generate_equity samples6 data6/validation.csv 18000 1024 825123 assets/preflop.csv > data6/validation.log 2>&1 &
wait
./generate_equity samples6 data6/test.csv 12000 2048 9371821 assets/preflop.csv > data6/test.log 2>&1 &
python3 tools/train_equity.py --data data6/train_*.csv --holdout data6/validation.csv --out assets/equity6_model.json.gz --report measurements/equity6_model.json --trees 24 --depth 18 --leaf 8
wait
echo 'BOT_9_EQUITY6_DONE'
