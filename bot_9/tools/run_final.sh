#!/bin/bash
#SBATCH --job-name=bot_9-final
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=bot_9-final-%j.log
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_9"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m unittest discover -s tests -v
python3 tools/final_evaluation.py
python3 tools/latency.py --calls 40000 --out measurements/latency_final_linux.json --progress
python3 tools/package_bot.py --output dist/bot_9.zip
echo 'BOT_9_FINAL_DONE'
