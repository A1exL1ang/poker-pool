#!/bin/bash
#SBATCH --job-name=bot_16-confirm
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=bot_16-confirm-%j.log
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_16"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
date -u
python3 -m unittest discover -s tests -v > benchmarks/remote_unit_tests.txt 2>&1
python3 benchmarks/tune.py --tables 256 --seed 210000 --fields mixed,tag,aggressive,adversarial --variants benchmarks/variants_v3.json --output benchmarks/tuning_v3.json
date -u
tar -czf bot_16-confirm-results.tgz benchmarks/tuning_v3.json benchmarks/remote_unit_tests.txt
