#!/bin/bash
set -euo pipefail
cd /home/acliang/poker-pool-extended-20261106/bot_11
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m tools.compact_equity
mv assets/equity.bin.z assets/equity_full_precision.bin.z
mv assets/equity_compact.bin.z assets/equity.bin.z
