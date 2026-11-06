#!/bin/bash
#SBATCH --job-name=bot_15-refine
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/refinement_job_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 -m training.tune_policy --stage refinement --tables 512 --workers 8 --output artifacts/policy_refinement.json
