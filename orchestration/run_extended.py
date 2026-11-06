"""Twelve additional independent teams; no build deadline; two helpers each."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import run_builders as base

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'orchestration/extended'
base.CONTROL=CONTROL
ASSIGNMENTS=[(9+i*2+j,m,e) for i,(m,e) in enumerate((('gpt-6.1-sol','high'),('gpt-6.1-sol','xhigh'),('gpt-6.1-sol','max'),('gpt-6-astra','high'),('gpt-6-astra','xhigh'),('gpt-6-astra','max'))) for j in range(2)]

def prompt(w):
    return f'''Build one independent poker bot.

Your strategic brief:
1. Read the complete RULES.txt for the contest rules.
2. Maximize the chance of winning the contest.
3. The game is standard no-limit Texas Hold'em.

Operational instructions authorized by the user:
- You are bot_{w['number']}, working ONLY in {w['workspace']}. Use this as the explicit working directory. No official starter kit is available; create your own local implementation and test environment and document assumptions. Never claim official interface compatibility or invent official payouts.
- There is NO one-hour deadline, NO orchestrator-imposed build timeout, and NO instruction to wrap up after a fixed duration. Take as long as useful. Pursue the strongest defensible result, including substantive research, implementation, training, experiments, validation and refinement where useful. A first working prototype is a milestone, not by itself a reason to finish. Choose your own methods. Finish when your own evidence supports a robust, well-tested submission and further work has diminishing returns; do not claim absolute optimality.
- Your team may spawn at most TWO helper agents total, using native collaboration tools. Both must work exclusively on this same bot. Reuse those helpers for additional subtasks if useful. Helpers must not spawn further agents or launch independent CLI sessions. Pass them these isolation/resource rules. Use the same model/effort as this head unless explicitly overridden by the user; service tier follows the root and must remain default/Fast OFF. Join/finish helpers before finalizing.
- STRICT INDEPENDENCE: never inspect, search, list, read, import, copy or modify any other bot, sibling workspace, parent project, earlier experiment, external private poker repository, or another team's logs/messages/artifacts. Never search above this workspace. Do not inspect global agent inventories or other conversations. Do not ask the orchestrator about other bots. Public documentation/research is allowed. Develop your own strategy from the brief.
- Use .control/ENGAGING.json for the authorized MIT Engaging connection. WSL Ubuntu-24.04 Linux SSH is required. User is actively using ORCD: ALL pre-existing allocations/jobs/nodes are OFF LIMITS. Never use srun --jobid for them; never consume, attach to, cancel, modify or stop them or their processes/sessions. Request your OWN new bounded jobs on mit_preemptable, with bot_{w['number']}-prefixed names and ONLY the private remote directory in ENGAGING.json. No heavy computation on login nodes. Prefer at most one active compute allocation per team, at most 8 CPUs/32GB or one GPU, walltime at most 2 hours per job; successive bounded jobs are allowed while useful. Do not request whole-node exclusive allocations. Keep a ledger of your own job IDs and stop only your own leftovers on completion. Do not stop or reconfigure the shared SSH master. If access expires, report it and continue locally; never solicit or print credentials.
- Configured Modal may be used for useful training in your own unique bot namespace; do not inspect others' projects, secrets or deployments. Keep each operation bounded and clean up only your own resources. Gameplay must have no online calls.
- This head uses {w['model']} / {w['reasoning']} and service_tier=default (Fast OFF). Do not change these settings or launch extra head processes. If a helper is unavailable, continue useful work yourself and report the issue.
- Update .control/STATUS.txt and read .control/INBOX.txt approximately every 5-10 minutes. Report current work, tests/evidence, own resource jobs, blockers and next steps. Preserve checkpoints and a runnable best candidate throughout long experiments. Keep metadata in .control; keep deliverable source, assets, README, tests, reproducible benchmark commands, measurements and limitations in your workspace.
- Before finishing, validate rule-critical mechanics, observation/action semantics, offline packaging and 1ms decision behavior; report measured overruns/fallbacks honestly. Separate training/tuning evidence from held-out evaluation. Never fabricate results. This is local experimentation, not contest submission. Do not push, publish or submit; the orchestrator will collect and independently evaluate your completed bot.
'''

def prepare():
    if (CONTROL/'status.json').exists():raise RuntimeError('Already prepared; no duplicate launch')
    (CONTROL/'logs').mkdir(parents=True)
    rules=(ROOT/'orchestration/rules.txt').read_bytes()
    conn=json.loads((ROOT/'orchestration/engaging_connection.json').read_text(encoding='utf-8-sig'))
    workers=[]
    for n,m,e in ASSIGNMENTS:
        path=ROOT/'.workers'/f'bot_{n}'
        if path.exists() or (ROOT/f'bot_{n}').exists():raise RuntimeError(f'Existing bot_{n}; refusing overwrite')
        path.mkdir(parents=True)
        subprocess.run(['git','init','-b',f'codex/bot-{n}',str(path)],check=True,capture_output=True)
        (path/'RULES.txt').write_bytes(rules)
        (path/'.gitignore').write_text('.control/\n.venv/\n__pycache__/\n*.pyc\n')
        (path/'.control').mkdir()
        c=dict(conn,remote_directory=f'~/poker-pool-extended-20261106/bot_{n}',partition='mit_preemptable',resource_policy='Only NEW bounded private jobs; every pre-existing user allocation/node/job is off limits.')
        (path/'.control/ENGAGING.json').write_text(json.dumps(c,indent=2))
        (path/'.control/INBOX.txt').write_text('No build time limit. Maintain strict team isolation; leave all existing ORCD jobs alone.\n')
        w=dict(number=n,model=m,reasoning=e,workspace=str(path),status='queued',helper_limit=2)
        (CONTROL/'logs'/f'bot_{n}.prompt.txt').write_text(prompt(w))
        workers.append(w)
    state=dict(status='prepared',created_at=base.now(),max_concurrent=12,build_deadline=None,fast_mode=False,service_tier='default',helpers_per_team=2,rules_sha256=hashlib.sha256(rules).hexdigest(),workers=workers,isolation='Independent repositories/contexts and strict no-cross-read instructions; not an OS read barrier.')
    base.save_json(CONTROL/'status.json',state)
    print(json.dumps({'status':'prepared','workers':12,'deadline':None}))

def run(recover=False):
    state=json.loads((CONTROL/'status.json').read_text())
    if state['status']!=('running' if recover else 'prepared'):raise RuntimeError('Unexpected status; refusing duplicate run')
    state.update(status='running',supervisor_pid=os.getpid(),last_poll_at=base.now())
    active={}
    codex=shutil.which('codex')
    if not codex:raise RuntimeError('Codex executable missing')
    if recover:
        for w in state['workers']:
            if w['status']=='running':active[w['number']]=(w,base.AttachedProcess(w),None,None)
    base.save_json(CONTROL/'status.json',state)
    for w in state['workers']:
        if w['status']!='queued':continue
        n=w['number'];out=open(CONTROL/'logs'/f'bot_{n}.jsonl','w',encoding='utf8');err=open(CONTROL/'logs'/f'bot_{n}.stderr.txt','w',encoding='utf8')
        args=[codex,'exec','--json','--color','never','--approve-for-me','--strict-config','-C',w['workspace'],'-m',w['model'],'-c',f'model_reasoning_effort="{w["reasoning"]}"','-c','service_tier="default"','-c','agents.enabled=true','-c','agents.max_concurrent_threads_per_session=2','-c',f'agents.default_subagent_model="{w["model"]}"','-c',f'agents.default_subagent_reasoning_effort="{w["reasoning"]}"','-o',str(CONTROL/'logs'/f'bot_{n}.final.txt'),'-']
        p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=out,stderr=err,text=True,encoding='utf8',cwd=w['workspace'],creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        w.update(status='running',started_at=base.now(),pid=p.pid,command=args[:-1])
        p.stdin.write(prompt(w));p.stdin.close()
        active[n]=(w,p,out,err)
        base.save_json(CONTROL/'status.json',state)
    while active:
        for n,(w,p,out,err) in list(active.items()):
            code=p.poll()
            if code is None:continue
            if out:out.close()
            if err:err.close()
            w.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=base.now())
            try:base.export_worker(w)
            except Exception as exc:w['export_error']=repr(exc)
            del active[n]
        state['last_poll_at']=base.now()
        base.save_json(CONTROL/'status.json',state)
        if active:time.sleep(10)
    state.update(status='finished',finished_at=base.now());base.save_json(CONTROL/'status.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','recover']);a=p.parse_args()
    prepare() if a.command=='prepare' else run(a.command=='recover')
