#!/bin/bash
#SBATCH --job-name=bot_10-equity
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=bot_10-equity-%j.log
set -eu
cd "$HOME/poker-pool-extended-20261106/bot_10"
mkdir -p .tmp .cache
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache"
python3 -m venv .venv
.venv/bin/pip --no-cache-dir -q install numpy scikit-learn
g++ -O3 -std=c++17 generate_equity.cpp -o generate_equity
./generate_equity 180000 96 10010 train.csv &
./generate_equity 12000 384 90010 validation.csv &
wait
.venv/bin/python train_equity.py --trees 24 --depth 18
