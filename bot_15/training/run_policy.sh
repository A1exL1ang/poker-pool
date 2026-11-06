#!/bin/bash
#SBATCH --job-name=bot_15-policy
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/policy_job_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 -m unittest discover -s tests
python3 -m training.tune_policy --tables 128 --workers 8
python3 -m benchmarks.latency --tables 100 --repeat 4 --output artifacts/latency_linux_initial.json
