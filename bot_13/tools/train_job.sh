#!/usr/bin/env bash
# Invoked by sbatch only, in bot_13's unique private remote directory.
#SBATCH --job-name=bot_13-train-v1
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$HOME/poker-pool-extended-20261106/bot_13/runs/train_v1/src"
printf 'bot_13 job %s started\n' "${SLURM_JOB_ID:?}"
python3 native/build.py
python3 tools/train_preflop.py --samples 200000 --seed 130613
python3 -m unittest discover -s tests -v
python3 tools/experiment.py --plan plans/training_v1.json --workers 8 --output ../results
printf 'bot_13 job %s finished\n' "$SLURM_JOB_ID"
