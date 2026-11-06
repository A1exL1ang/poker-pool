#!/bin/bash
# Run ONLY in a newly requested bot_6 Slurm allocation, never a login node.
set -euo pipefail
cd "$HOME/poker-pool-20261106-054844/bot_6"
test -n "${SLURM_JOB_ID:-}"
mkdir -p benchmarks
python3 --version
python3 -m unittest discover -s tests -v > benchmarks/linux_tests.txt 2>&1
python3 -m tools.benchmark --iterations 30000 --output benchmarks/linux_latency.json
python3 -m tools.simulate --tables 300 --seed 36006 --field tight --enforce-timeout --output benchmarks/linux_tight.json
python3 -m tools.simulate --tables 300 --seed 46006 --field calling --enforce-timeout --output benchmarks/linux_calling.json
python3 -m tools.simulate --tables 300 --seed 56006 --field maniac --enforce-timeout --output benchmarks/linux_maniac.json
python3 -m tools.simulate --tables 100 --seed 66006 --field self --enforce-timeout --output benchmarks/linux_self.json
printf 'BOT_6_VALIDATION_COMPLETE\n'
