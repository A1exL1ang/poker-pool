#!/bin/bash
#SBATCH --job-name=bot_16-final-eval
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=bot_16-final-%j.log
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_16"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
date -u
python3 benchmarks/final_suite.py
date -u
tar -czf bot_16-final-results.tgz benchmarks/heldout_*.json benchmarks/freeze.json benchmarks/equity_heldout.json benchmarks/latency_final_linux.json benchmarks/package_check.json benchmarks/final_unit_tests.txt MEASUREMENTS.md dist/bot_16.zip
