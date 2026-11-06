"""Assemble pure CFR net v9 with unchanged pinned submission runtime/postflop nets."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
source=ROOT/'baseline_repo/vibe_coding_exp/claude_exp_1'
target=ROOT/'baseline_net'
target.mkdir(exist_ok=True)
shutil.copytree(source/'submission/nlbot',target/'nlbot',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for model in ('policy_v9','postflop_v6s','postflop3_v1'):
    shutil.copytree(source/'models'/model,target/'models'/model,dirs_exist_ok=True)
config={'kind':'nlbot','policy':'policy_v9','solver_max_bb':1000.0,'postflop_policy':'postflop_v6s','postflop_max_bb':1000.0,'postflop3_policy':'postflop3_v1','heuristic_params':{}}
(target/'submission_config.json').write_text(json.dumps(config,indent=2))
manifest={'repository':'https://github.com/A1exL1ang/poker-prep','commit':'a3d75f7c736da8c69e474ceed7e189682f9e6c7e','config':config,'description':'Pure CFR-distilled preflop net v9; no PPO finetuning; original default heuristic fallback remains. Runtime copied from pinned submission, selected weights copied from pinned models.','files':{p.relative_to(target).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}}
(ROOT/'net_baseline_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'files':len(manifest['files']),'config':config}))
