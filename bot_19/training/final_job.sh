#!/bin/bash
#SBATCH --job-name=bot_19_final_validation
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=.control/final-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=19
python3 -m unittest discover -s tests -v > .control/final-tests.log 2>&1
g++ -O3 -std=c++17 -fopenmp native/equity.cpp -o native/equity
python3 training/validate_native.py ./native/equity --output measurements/native_final_validation.json
python3 training/validate_fast.py --cases 20000 --output measurements/equity_codegen_final_linux.json
python3 benchmarks/final_evaluate.py --games 5400 --baseline-games 1800 --workers 8
python3 benchmarks/run.py --games 1200 --split heldout --offset 20000 --pools mixed,tight,pressure,calling,mirror,diverse --payouts chip,top3,linear,top2,satellite --schedules local,fast,slow --workers 8 --params assets/selected_params.json --enforce --output measurements/final_enforced.json
python3 benchmarks/latency.py --tables 240 --repeat 3 --params assets/selected_params.json --output measurements/final_latency.json
python3 benchmarks/stress.py --iterations 3000 --params assets/selected_params.json --output measurements/final_stress.json
python3 benchmarks/report_results.py
python3 build_package.py
