#!/bin/bash
#SBATCH --job-name=bot_2_latency
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:03:00
#SBATCH --output=bot_2_latency_%j.log
set -euo pipefail
cd "$HOME/poker-pool-20261106-054844/bot_2"
python3 --version
python3 -m unittest -q tests.test_bot > benchmarks/linux_tests.txt 2>&1
python3 -m tools.benchmark latency --iterations 50000 --output benchmarks/linux_latency.json > benchmarks/linux_latency.stdout
