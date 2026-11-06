#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --job-name=bot_14_confirm
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:45:00
#SBATCH --output=bot_14_confirm-%j.out
#SBATCH --error=bot_14_confirm-%j.err
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_14"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m unittest discover -s tests -q
python3 tools/package.py --verify --report measurements/package_draft_validation.json
python3 tools/confirm.py --configs measurements/confirmation_configs.json --tables 1000 --sensitivity-tables 500 --workers 8 --seed 144000 --output measurements/confirmation.json
