#!/bin/bash
#SBATCH --job-name=bot_9-refine
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=bot_9-refine-%j.log
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_9"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 tools/tune.py --tables 768 --workers 8 --candidates tools/stage2_candidates.json --suites tools/stage2_suites.json --out measurements/tuning_stage2.json
echo 'BOT_9_STAGE2_DONE'
