#!/bin/bash
#SBATCH --job-name=bot_19_refine
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=.control/refine-%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=19
.venv-training/bin/python training/refine.py
.venv-training/bin/python training/export_fast.py
.venv-training/bin/python training/validate_fast.py
./native/equity generate training/data/base.txt training/data/final_equity_test.bin 30000 8192 1900205
.venv-training/bin/python training/evaluate_model.py --test training/data/final_equity_test.bin --report measurements/equity_final_heldout.json
