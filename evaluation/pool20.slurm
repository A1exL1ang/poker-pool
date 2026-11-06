#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --job-name=pool20-production
#SBATCH --array=0-23
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=evaluation
python3 evaluation/verify_pool20.py
python3 evaluation/verify_bundle.py --net
python3 evaluation/run_pool20.py --shard "$SLURM_ARRAY_TASK_ID" --shards 24 --repetitions 4 --output "evaluation/pool20_results/production_${SLURM_ARRAY_TASK_ID}.jsonl"
