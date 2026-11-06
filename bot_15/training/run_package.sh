#!/bin/bash
#SBATCH --job-name=bot_15-package
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/package_job_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 -m training.summarize
python3 -m training.package_bot
