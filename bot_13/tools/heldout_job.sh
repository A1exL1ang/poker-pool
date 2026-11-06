#!/usr/bin/env bash
#SBATCH --job-name=bot_13-heldout-v4
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HOME/poker-pool-extended-20261106/bot_13/runs/heldout_v4/src"
printf 'bot_13 job %s started\n' "${SLURM_JOB_ID:?}"
mkdir -p ../audit
python3 native/build.py
python3 tools/test_suite.py --output ../audit/test_summary.json
python3 tools/package.py
python3 tools/validate_package.py --output ../audit/package_validation.json
python3 tools/experiment.py --plan plans/heldout_v4.json --workers 8 --output ../results
python3 -m lab.benchmark --tables 32 --seed-start 1030000 --lineup mixed --enforce-deadline --observations ../audit/final_observations.jsonl.gz --output ../audit/timing_capture.json --progress-every 8
python3 tools/latency.py --corpus ../audit/final_observations.jsonl.gz --repeats 5 --output ../audit/serial_replay.json
python3 tools/stress_timing.py --iterations 2000 --output ../audit/stress_timing.json
printf 'bot_13 job %s finished\n' "$SLURM_JOB_ID"
