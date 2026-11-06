#!/bin/bash
#SBATCH --job-name=poker-pool-eval-pilot
#SBATCH --partition=mit_preemptable
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=pilot-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HOME/poker-pool-20261106-054844/evaluation/run_v1"
python3 evaluation/run_matches.py --pilot --seeds 1 --shard "$SLURM_ARRAY_TASK_ID" --shards 4 --output "evaluation/results/pilot_${SLURM_ARRAY_TASK_ID}.jsonl"
