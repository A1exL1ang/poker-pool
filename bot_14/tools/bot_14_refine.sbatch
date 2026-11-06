#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --job-name=bot_14_refine
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:45:00
#SBATCH --output=bot_14_refine-%j.out
#SBATCH --error=bot_14_refine-%j.err
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_14"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p build assets measurements .control
python3 -m unittest tests.test_pushfold -v
python3 tools/train_pushfold.py --assets assets/preflop.json --output assets/pushfold.json --iterations 20000 --checkpoint .control/bot_14_pushfold_checkpoint.json
g++ -std=c++17 -O3 -fopenmp tools/postflop_dataset.cpp -o build/postflop_dataset
./build/postflop_dataset --rows 1200000 --trials 256 --balanced 1 --seed 140220 --ranks build/preflop_ranks.txt --output build/postflop_train_v2.bin
python3 tools/train_equity.py --train build/postflop_train_v2.bin --validation build/postflop_validation.bin --derived --iterations 300 --leaves 63 --output assets/equity_model_v2.json --report measurements/equity_validation_v2.json
python3 tools/benchmark.py --tables 500 --seed 141000 --field mixed --workers 8 --output measurements/v1_development_mixed.json
python3 tools/benchmark.py --tables 500 --seed 141000 --field rollout --workers 8 --output measurements/v1_development_rollout.json
python3 tools/benchmark.py --tables 500 --seed 141000 --field aggressive --workers 8 --output measurements/v1_development_aggressive.json
cp assets/equity_model.json assets/equity_model_v1.json
cp assets/equity_model_v2.json assets/equity_model.json
python3 tools/benchmark.py --tables 500 --seed 141000 --field mixed --workers 8 --output measurements/v2_development_mixed.json
python3 tools/benchmark.py --tables 500 --seed 141000 --field rollout --workers 8 --output measurements/v2_development_rollout.json
python3 tools/benchmark.py --tables 500 --seed 141000 --field aggressive --workers 8 --output measurements/v2_development_aggressive.json
