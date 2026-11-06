"""Dedicated, bounded local decision latency replay; no contest claims.

python tools/latency.py --calls 20000 --out measurements/latency.json
python tools/latency.py --calls 20000 --model-path assets/equity_model12.bin

Collects reachable observations from separate synthetic mixed/mirror tables,
then replays them in original table order with a fresh agent per table/seat.
Both wall and process clocks are measured; no outliers or timer overhead are
removed. Replay observations follow the un-enforced collection trajectory.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import make_baseline
from bot import Agent
from engine import tournament
from evaluator import evaluate

BLINDS = [(10,20),(15,30),(25,50),(40,80),(60,120),(100,200),(150,300),(250,500)]
PAYOUTS = [.5,.3,.2,0.,0.,0.,0.,0.]
MIXED = ['station','nit','loose','maniac','random','value','value']


def distribution(values):
    values = sorted(values)
    n = len(values)
    def q(fraction):
        if not n:
            return None
        at = (n-1)*fraction
        low = int(at)
        return values[low] + (values[min(low+1,n-1)]-values[low])*(at-low)
    over = sum(v > 1_000_000 for v in values)
    return dict(n=n, mean_ns=statistics.fmean(values) if n else None,
                p50_ns=q(.5),p95_ns=q(.95),p99_ns=q(.99),p999_ns=q(.999),
                max_ns=values[-1] if n else None,over_1ms=over,
                over_1ms_rate=over/n if n else None)


class Recorder:
    def __init__(self, agent, seat, records):
        self.agent,self.seat,self.records = agent,seat,records

    @property
    def errors(self):
        return self.agent.errors

    def act(self, observation):
        action = self.agent.act(observation)
        # engine.observation creates fresh immutable-for-our-purpose snapshots.
        self.records.append((self.seat,observation,action))
        return action


def collect(lineup, target, *, seed_start=7000, model_path=None, max_tables=400, max_hands=200):
    if not 0 <= target <= 100000:
        raise ValueError('Collection is bounded at 100,000 recorded decisions')
    tables=[]
    collected=errors=internal=0
    started=time.perf_counter()
    for offset in range(max_tables):
        if collected >= target:
            break
        seed=seed_start+offset
        records=[]
        focal=offset%8
        agents=[]
        seeds={}
        villain=0
        for seat in range(8):
            agent_seed=seed*101+seat+9
            if lineup=='mirror' or seat==focal:
                seeds[seat]=agent_seed
                agents.append(Recorder(Agent(seed=agent_seed,model_path=model_path),seat,records))
            else:
                agents.append(make_baseline(MIXED[(villain+offset//8)%7],seed=agent_seed))
                villain+=1
        result=tournament(agents,seed=seed,payouts=PAYOUTS,blind_schedule=BLINDS,
                          evaluator=evaluate,max_hands=max_hands,
                          enforce_time_limit=False,record_latencies=False)
        errors+=result['errors']
        internal+=sum(getattr(a,'errors',0) for a in agents)
        records=records[:target-collected]
        tables.append(dict(seed=seed,agent_seeds=seeds,records=records))
        collected+=len(records)
    return tables,dict(lineup=lineup,requested_decisions=target,recorded_decisions=collected,
        tables=len(tables),seed_start=seed_start,engine_errors=errors,internal_fallbacks=internal,
        wall_seconds=time.perf_counter()-started,limit_enforcement=False,
        synthetic_payouts=PAYOUTS,synthetic_blinds=BLINDS,max_hands=max_hands)


STARTUP = r'''
import json,pathlib,sys,time
sys.path.insert(0,sys.argv[1])
p=time.perf_counter_ns();c=time.process_time_ns()
from bot import Agent
import_cpu=time.process_time_ns()-c;import_wall=time.perf_counter_ns()-p
path=None if sys.argv[2]=='-' else sys.argv[2]
p=time.perf_counter_ns();c=time.process_time_ns()
agent=Agent(seed=901,model_path=path)
init_cpu=time.process_time_ns()-c;init_wall=time.perf_counter_ns()-p
o=dict(hero=3,hole=[48,49],board=[],stacks=[2000,1990,1980,2000,2000,2000,2000,2000],
 committed=[0,10,20,0,0,0,0,0],contributions=[0,10,20,0,0,0,0,0],folded=[False]*8,
 all_in=[False]*8,button=0,street=0,pot=30,to_call=20,min_raise_to=40,max_raise_to=2000,
 can_raise=True,bb=20,sb=10,hand_no=0,payouts=None,history=[])
p=time.perf_counter_ns();c=time.process_time_ns()
a=agent.act(o)
act_cpu=time.process_time_ns()-c;act_wall=time.perf_counter_ns()-p
print(json.dumps(dict(import_wall_ns=import_wall,import_process_ns=import_cpu,
 agent_init_wall_ns=init_wall,agent_init_process_ns=init_cpu,
 first_preflop_call_wall_ns=act_wall,first_preflop_call_process_ns=act_cpu,
 first_action=a,internal_fallbacks=agent.errors,model_trees=len(agent.model.trees))))
'''


def startup_measurement(model_path=None):
    started=time.perf_counter_ns()
    result=subprocess.run([sys.executable,'-I','-S','-c',STARTUP,str(ROOT),str(model_path) if model_path else '-'],
                          cwd=ROOT,text=True,capture_output=True,check=True,timeout=90)
    report=json.loads(result.stdout)
    report['subprocess_wall_including_launch_ns']=time.perf_counter_ns()-started
    return report


def valid_action(action,o):
    if not isinstance(action,dict) or action.get('action') not in ('fold','check','call','raise'):
        return False
    if action['action']=='check' and o['to_call']:
        return False
    if action['action']=='raise':
        amount=action.get('amount')
        if isinstance(amount,bool) or not isinstance(amount,int) or not o['can_raise']:
            return False
        if not max(o['committed'])<amount<=o['max_raise_to']:
            return False
        if amount<o['min_raise_to'] and amount!=o['max_raise_to']:
            return False
    return True


def replay(tables,model_path=None):
    samples=[]
    mismatches=invalid=exceptions=internal=0
    wall_fallbacks=process_fallbacks=0
    init_wall=[]
    init_cpu=[]
    for table in tables:
        agents={}
        for seat,seed in table['agent_seeds'].items():
            start_wall=time.perf_counter_ns()
            start_cpu=time.process_time_ns()
            agents[seat]=Agent(seed=seed,model_path=model_path)
            init_cpu.append(time.process_time_ns()-start_cpu)
            init_wall.append(time.perf_counter_ns()-start_wall)
        first=set()
        for seat,o,expected in table['records']:
            agent=agents[seat]
            before=agent.errors
            exception=None
            start_wall=time.perf_counter_ns()
            start_cpu=time.process_time_ns()
            try:
                action=agent.act(o)
            except Exception as error:
                exception=type(error).__name__
                action=agent.fallback(o)
            cpu=time.process_time_ns()-start_cpu
            wall=time.perf_counter_ns()-start_wall
            fallback=agent.errors-before
            bad=not valid_action(action,o)
            invalid+=bad
            exceptions+=exception is not None
            internal+=fallback
            mismatches+=action!=expected
            wall_fallbacks+=bool(wall>1_000_000 or bad or exception or fallback)
            process_fallbacks+=bool(cpu>1_000_000 or bad or exception or fallback)
            samples.append(dict(wall_ns=wall,process_ns=cpu,street=o['street'],
                                first_call=seat not in first,internal_fallbacks=fallback,
                                invalid=bad,exception=exception))
            first.add(seat)
    def group(predicate):
        selected=[s for s in samples if predicate(s)]
        return dict(wall=distribution([s['wall_ns'] for s in selected]),
                    process=distribution([s['process_ns'] for s in selected]))
    nonzero=Counter(s['process_ns'] for s in samples if s['process_ns']>0)
    return dict(all=group(lambda s:True),preflop=group(lambda s:s['street']==0),
                postflop=group(lambda s:s['street']>0),
                warm_calls=group(lambda s:not s['first_call']),
                table_first_calls=group(lambda s:s['first_call']),
                by_street={str(n):group(lambda s,n=n:s['street']==n) for n in range(4)},
                agent_init_warm_process=dict(wall=distribution(init_wall),process=distribution(init_cpu)),
                action_mismatches=mismatches,invalid_actions=invalid,uncaught_exceptions=exceptions,
                internal_safety_fallbacks=internal,
                process_clock_observed_nonzero_ns=nonzero.most_common(8),
                would_trigger_wall_limit_or_error_fallback=wall_fallbacks,
                would_trigger_process_limit_or_error_fallback=process_fallbacks)


def run(args):
    if not 1<=args.calls<=100000:
        raise ValueError('--calls must be 1..100000')
    model_path=str(Path(args.model_path).resolve()) if args.model_path else None
    selected=Path(model_path) if model_path else ROOT/'assets/equity_model.bin'
    if not selected.exists() and model_path is None:
        selected=ROOT/'assets/equity_model.json.gz'
    artifacts={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
               for name in ('bot.py','model.py','features.py','evaluator.py','equity.py','assets/preflop.csv')}
    artifacts['selected_model']=hashlib.sha256(selected.read_bytes()).hexdigest()
    report=dict(benchmark='independent-local-latency-replay-v1',
        runtime=dict(python=sys.version,platform=platform.platform()),
        clocks={name:dict(implementation=(info:=time.get_clock_info(name)).implementation,
                         resolution_seconds=info.resolution,monotonic=info.monotonic,
                         adjustable=info.adjustable) for name in ('perf_counter','process_time')},
        config=dict(calls=args.calls,model_path=str(selected),seed_start=args.seed_start,
                    max_tables=args.max_tables,max_hands=args.max_hands),
        startup=startup_measurement(model_path),artifact_sha256=artifacts,lineups={},
        semantics=['Timers surround the full Agent.act call; no timer overhead or outliers removed.',
            'Wall timer also includes the nested process clock calls and process scheduling.',
            'Windows process_time can update in coarse ~15.625ms ticks; raw process overrun counts may not indicate individual slow calls. Observed nonzero tick sizes are retained.',
            'Collection uses no timing enforcement; replay fallback counts describe what a 1ms wrapper would do on this fixed trajectory.',
            'New table-local agents and original observation order preserve state; mismatches are reported.',
            'Synthetic lineups, payouts and blinds; hardware-specific local evidence, not official compatibility or performance.'])
    targets={'mixed':(args.calls+1)//2,'mirror':args.calls//2}
    started=time.perf_counter()
    for index,(lineup,target) in enumerate(targets.items()):
        tables,collection=collect(lineup,target,seed_start=args.seed_start+index*1000,
            model_path=model_path,max_tables=args.max_tables,max_hands=args.max_hands)
        report['lineups'][lineup]=dict(collection=collection,replay=replay(tables,model_path))
        if args.progress:
            data=report['lineups'][lineup]['replay']
            print(f'{lineup}: {collection["recorded_decisions"]} decisions, wall>1ms={data["all"]["wall"]["over_1ms"]}, process>1ms={data["all"]["process"]["over_1ms"]}',file=sys.stderr)
    report['total_wall_seconds']=time.perf_counter()-started
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--calls',type=int,default=20000)
    parser.add_argument('--model-path')
    parser.add_argument('--seed-start',type=int,default=7000)
    parser.add_argument('--max-tables',type=int,default=400)
    parser.add_argument('--max-hands',type=int,default=200)
    parser.add_argument('--progress',action='store_true')
    parser.add_argument('--out',default='measurements/latency.json')
    args=parser.parse_args()
    report=run(args)
    path=(ROOT/args.out).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Latency report must stay in this bot workspace')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(startup=report['startup'],lineups={name:data['replay']['all'] for name,data in report['lineups'].items()},total_wall_seconds=report['total_wall_seconds']),indent=2))


if __name__=='__main__':
    main()
