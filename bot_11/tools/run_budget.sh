#!/bin/bash
set -euo pipefail
cd /home/acliang/poker-pool-extended-20261106/bot_11
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cp assets/equity.bin.z assets/equity_64.bin.z
python3 -m tools.model_budget --source assets/equity_64.bin.z
# Generate a third, independent final test without inspecting model accuracy
# on it until after the model/policy choice is frozen.
training/generate_equity --mode postflop --states 15000 --rollouts 16384 --seed 1198800 --threads 2 --out training/final_test.bin
