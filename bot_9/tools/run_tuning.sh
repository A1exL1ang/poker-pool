#!/bin/bash
#SBATCH --job-name=bot_9-policy
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=bot_9-policy-%j.log
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_9"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 tools/compact_model.py --source assets/equity6_model.json.gz --out assets/equity_model.bin --trees 12 --holdout data6/validation.csv --rows 18000 --report measurements/equity6_compact_validation_linux.json
python3 tools/tune.py --tables 128 --workers 8 --out measurements/tuning_selection.json
echo 'BOT_9_POLICY_DONE'
