#!/bin/bash
#SBATCH --job-name=poker-pool-pure-net
#SBATCH --partition=mit_preemptable
#SBATCH --array=0-23
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=net-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HOME/poker-pool-20261106-054844/evaluation/run_v1"
python3 evaluation/verify_bundle.py --net
python3 evaluation/run_matches.py --baseline net --seeds 64 --shard "$SLURM_ARRAY_TASK_ID" --shards 24 --output "evaluation/net_results/production_${SLURM_ARRAY_TASK_ID}.jsonl"
