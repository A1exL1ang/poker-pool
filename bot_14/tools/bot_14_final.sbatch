#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --job-name=bot_14_final
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:45:00
#SBATCH --output=bot_14_final-%j.out
#SBATCH --error=bot_14_final-%j.err
set -euo pipefail
cd "$HOME/poker-pool-extended-20261106/bot_14"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p build measurements
python3 -m unittest discover -s tests -q
if test -f measurements/locked_candidate.json; then
    python3 tools/package.py --verify --report measurements/package_validation.json
else
    python3 tools/package.py --verify --lock measurements/locked_candidate.json --config "$(cat measurements/selected_config.json)" --report measurements/package_validation.json
fi
./build/postflop_dataset --rows 30000 --trials 4096 --seed 2014014 --threads 8 --ranks build/preflop_ranks.txt --output build/postflop_heldout.bin
python3 tools/train_equity.py --evaluate assets/equity_model.json --validation build/postflop_heldout.bin --report measurements/equity_heldout.json
./build/equity_assets --trials 20000 --multiway-trials 32 --threads 8 --seed 28914014 --output build/preflop_heldout.json
python3 tools/audit_pushfold.py --policy assets/pushfold.json --equity build/preflop_heldout.json --output measurements/pushfold_heldout.json
export OMP_NUM_THREADS=1
python3 tools/final_evaluate.py --lock measurements/locked_candidate.json --tables 2000 --sensitivity-tables 500 --workers 8 --seed 1014000 --timeout-ms 1 --output measurements/heldout_tournaments.json
python3 tools/latency.py --generate --corpus measurements/latency_final_corpus.jsonl.gz --tables 40 --per-street 1200 --seed 9014014 --repeat 20 --output measurements/latency_final_linux.json
python3 tools/latency.py --corpus measurements/latency_final_corpus.jsonl.gz --repeat 5 --config '{"native":false}' --output measurements/latency_portable_linux.json
python3 tools/package.py --verify --report measurements/package_validation.json
