#!/bin/bash
#SBATCH --job-name=bot_8_validation
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:04:00
#SBATCH --output=evidence/linux_job_%j.log
set -euo pipefail
cd "$HOME/poker-pool-20261106-054844/bot_8"
mkdir -p evidence
python3 -m unittest discover -s tests -v > evidence/linux_tests.txt 2>&1
python3 -m tools.benchmark --tables 40 --repeat 3 --output evidence/linux_latency.json
python3 -m tools.verify_evaluator > evidence/linux_evaluator.json
python3 -c 'import hashlib,json,pathlib; files=[pathlib.Path("agent.py"),*sorted(pathlib.Path("kestrel").glob("*.py")),pathlib.Path("kestrel/preflop.json")]; print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))' > evidence/linux_source_hashes.json
