#!/bin/bash
# This script may only be submitted in the bot_19 private remote directory.
#SBATCH --job-name=bot_19_equity
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:50:00
#SBATCH --output=.control/equity-%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=19
mkdir -p .control assets measurements training/data
g++ -O3 -std=c++17 -fopenmp native/equity.cpp -o native/equity
python3 training/validate_native.py ./native/equity --output measurements/native_validation.json
python3 -m venv .venv-training
.venv-training/bin/python -m pip install --disable-pip-version-check --timeout 60 --retries 2 'numpy<3' 'xgboost<4'
./native/equity base training/data/base.txt 100000 1900101
./native/equity preflop training/data/base.txt assets/preflop.json 12000 1900102
./native/equity generate training/data/base.txt training/data/train.bin 600000 192 1900103
./native/equity generate training/data/base.txt training/data/validation.bin 45000 1024 1900104
./native/equity generate training/data/base.txt training/data/heldout.bin 24000 4096 1900105
.venv-training/bin/python training/train.py --train training/data/train.bin --valid training/data/validation.bin --trees 192 --depth 6 --threads 8
.venv-training/bin/python training/export_fast.py
.venv-training/bin/python training/validate_fast.py
.venv-training/bin/python training/evaluate_model.py --test training/data/heldout.bin
