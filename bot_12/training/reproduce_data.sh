#!/usr/bin/env bash
# Reproduce final data using at most eight workers; run in a bounded allocation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p training/data
g++ -O3 -std=c++17 -o training/generate training/generate.cpp
training/generate order training/data/order.txt 200000 1200201
training/generate preflop training/data/preflop.csv 150000 1200202 training/data/order.txt &
pids=("$!")
for k in 0 1 2 3 4 5; do
  training/generate postflop "training/data/train${k}.csv" 1024 "$((1200300+k))" training/data/order.txt 50000 >"training/data/train${k}.log" 2>&1 &
  pids+=("$!")
done
training/generate postflop training/data/heldout.csv 4096 1200400 training/data/order.txt 15000 >training/data/heldout.log 2>&1 &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" != 0 ]]; then exit 1; fi
printf 'BOT_12_DATA_COMPLETE\n'
