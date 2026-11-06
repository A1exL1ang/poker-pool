#!/usr/bin/env bash
#SBATCH --job-name=bot_13-selection-v3
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HOME/poker-pool-extended-20261106/bot_13/runs/selection_v3/src"
printf 'bot_13 job %s started\n' "${SLURM_JOB_ID:?}"
python3 native/build.py
python3 tools/train_pushfold.py --samples 100000 --threads 8 --seed 131313 --iterations 20000 --target-gap .002
python3 -m unittest discover -s tests -q
python3 tools/experiment.py --plan plans/selection_v3.json --workers 8 --output ../results
printf 'bot_13 job %s finished\n' "$SLURM_JOB_ID"
