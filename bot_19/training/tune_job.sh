#!/bin/bash
#SBATCH --job-name=bot_19_policy_tune
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=.control/tune-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=19
python3 -m unittest discover -s tests -q
python3 benchmarks/tune.py --candidates 28 --games 240 --finalists 6 --validation-games 1200 --workers 8
