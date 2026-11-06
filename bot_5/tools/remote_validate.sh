#!/bin/bash
#SBATCH --job-name=bot_5_validate
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:08:00
#SBATCH --output=remote_validation_%j.log
set -euo pipefail
cd "$HOME/poker-pool-20261106-054844/bot_5"
mkdir -p evidence
date -u
python3 --version
python3 -m unittest discover -s tests -v > evidence/linux_tests.txt 2>&1
python3 tools/benchmark.py --tables 64 --repeats 3 --seed 181000 --output evidence/linux_latency.json
python3 simulator.py --tables 256 --seed 182000 --enforce-timeout --output evidence/linux_timeout_simulation.json
date -u
