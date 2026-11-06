#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --job-name=bot_14_tune
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:45:00
#SBATCH --output=bot_14_tune-%j.out
#SBATCH --error=bot_14_tune-%j.err
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_14"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p pokerbot/_native measurements build
g++ -std=c++17 -O3 -fPIC -shared -static-libstdc++ -static-libgcc tools/runtime_equity.cpp -o pokerbot/_native/libb14_equity.so
# Refresh the optional probe using a Linux executable at its documented path.
g++ -std=c++17 -O3 -pthread tools/equity_assets.cpp -o tools/equity_assets
python3 -m unittest discover -s tests -q
python3 tools/latency.py --repeat 8 --config '{"native_mix":1}' --output measurements/latency_native_model_linux_development.json
python3 tools/latency.py --repeat 5 --config '{"native_mix":0.5}' --output measurements/latency_hybrid_linux_development.json
python3 tools/tune.py --tables 500 --workers 8 --seed 142000 --output measurements/tuning.json
