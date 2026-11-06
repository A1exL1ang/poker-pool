#!/bin/bash
#SBATCH --job-name=bot_15-heldout
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/final_job_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 -m unittest discover -s tests
python3 -m benchmarks.heldout --tables 1000 --workers 8 --output artifacts/heldout.json
python3 -m benchmarks.heldout --tables 200 --workers 8 --enforce-1ms --output artifacts/heldout_enforced_subset.json
python3 -m benchmarks.fuzz --tables 1000 --seed 7410001 --output artifacts/integration_fuzz_final.json
python3 -m benchmarks.latency --tables 200 --repeat 5 --output artifacts/latency_linux_final.json
