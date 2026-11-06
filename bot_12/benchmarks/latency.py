"""Serial decision timing on actual local observations and artificial ICM stress.

Run when other work is quiet, after freezing runtime assets:
  python benchmarks/latency.py --refresh-corpus --output measurements/latency.json
Separately enable in-game 1ms fallbacks and a same-policy unbounded control:
  python benchmarks/latency.py --enforce-deadline --deadline-tables 256 \
      --output measurements/latency_deadline.json

Agent construction, asset imports, observation construction and JSON transport
are outside the act() timer. OS scheduling pauses remain in wall-clock results.
No CPU, process, job, node or allocation inventory is queried.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
from time import perf_counter,perf_counter_ns,process_time_ns
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent import Agent,DEFAULTS
from baselines import make_lineup
from benchmarks.evaluate import PAYOUTS,SCHEDULES,summary,balanced_hero_seat
from icm import _hero_equity
from simulator import run_tournament

STREETS=("preflop","flop","turn","river")
SOURCE_FILES=("agent.py","equity.py","icm.py","assets/equity_predict.py",
              "assets/equity_model.json","assets/preflop.json","assets/preflop_narrow.json",
              "assets/rank_scores.zlib","assets/__init__.py","baselines.py","simulator.py")


def _local_path(value):
    path=(ROOT/value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("benchmark paths must remain inside this bot workspace")
    return path


def _hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in SOURCE_FILES if (ROOT/name).exists()}


def _latencies(values):
    values=sorted(values)
    def q(p):
        return values[min(len(values)-1,int((len(values)-1)*p))]/1000 if values else 0.
    return dict(decisions=len(values),p50_us=q(.5),p95_us=q(.95),p99_us=q(.99),
                p999_us=q(.999),max_us=q(1),overruns_1ms=sum(v>1_000_000 for v in values),
                overrun_fraction=sum(v>1_000_000 for v in values)/max(1,len(values)))


def _valid(obs,action):
    if not isinstance(action,dict) or action.get("action") not in obs["legal_actions"]:
        return False
    if action["action"]!="raise":
        return True
    amount=action.get("amount")
    return (type(amount) is int and obs["current_bet"]<amount<=obs["max_raise_to"]
            and (amount>=obs["min_raise_to"] or amount==obs["max_raise_to"]))


class Timing:
    def __init__(self):
        self.wall={street:[] for street in ("all",*STREETS)}
        self.cpu={street:[] for street in ("all",*STREETS)}
        self.fallbacks=0
        self.invalid=0
        self.icm_cache_misses=0

    def act(self,bot,obs,record=True):
        fallback_before=bot.fallbacks
        misses_before=_hero_equity.cache_info().misses
        cpu_start=process_time_ns()
        wall_start=perf_counter_ns()
        action=bot.act(obs)
        wall=perf_counter_ns()-wall_start
        cpu=process_time_ns()-cpu_start
        if record:
            for key in ("all",obs["street"]):
                self.wall[key].append(wall)
                self.cpu[key].append(cpu)
            self.fallbacks+=bot.fallbacks-fallback_before
            self.invalid+=not _valid(obs,action)
            self.icm_cache_misses+=_hero_equity.cache_info().misses-misses_before
        return action

    def report(self):
        return dict(wall={k:_latencies(v) for k,v in self.wall.items()},
                    cpu={k:_latencies(v) for k,v in self.cpu.items()},
                    process_cpu_samples_nonzero=sum(v>0 for v in self.cpu["all"]),
                    total_measured_act_wall_ms=sum(self.wall["all"])/1e6,
                    total_measured_act_process_cpu_ms=sum(self.cpu["all"])/1e6,
                    internal_fallbacks=self.fallbacks,invalid_actions=self.invalid,
                    icm_cache_misses=self.icm_cache_misses,
                    cpu_timing_note="Process CPU intervals include two wall-clock marker calls. OS CPU accounting can be coarse (many zero samples on Windows); CPU per-act percentiles do not establish sub-ms precision. Wall intervals bracket act only.")


def collect_corpus(*,tables=48,seed_start=730000,max_hands=200,params=None):
    records=[]
    table_rows=[]
    scenarios=("mixed","selfplay","shove")
    started=perf_counter()
    for index in range(tables):
        scenario=scenarios[index%len(scenarios)]
        seed=seed_start+index
        seat=balanced_hero_seat(index)
        bot=Agent(seed=seed^120012,params=params)
        opponents=make_lineup(scenario,seed=seed*17)
        random.Random(seed^0x12B0).shuffle(opponents)
        opponents.insert(seat,bot)
        def observe(obs):
            if obs["hero"]==seat:
                records.append(dict(table=index,seed=seed,hero_seat=seat,scenario=scenario,observation=obs))
        result=run_tournament(opponents,seed,max_hands=max_hands,
                              blind_schedule=SCHEDULES["doubling"],payouts=PAYOUTS["top3"],observer=observe)
        table_rows.append(dict(table=index,seed=seed,hero_seat=seat,scenario=scenario,
                               decisions=result["decisions"][seat],hands=result["hands"],
                               invalid=result["invalid_actions"][seat],exceptions=result["exceptions"][seat],
                               internal_fallbacks=bot.fallbacks))
    if not records:
        raise ValueError("corpus collection produced no hero decisions")
    metadata=dict(kind="actual_local_hero_observations",tables=tables,seed_start=seed_start,
                  max_hands=max_hands,synthetic_payouts=PAYOUTS["top3"],
                  synthetic_blind_schedule=SCHEDULES["doubling"],scenarios=list(scenarios),
                  hero_seat_rotation=True,seconds=perf_counter()-started,source_hashes=_hashes(),
                  params=dict(DEFAULTS,**(params or {})),table_rows=table_rows,
                  observations=len(records),
                  street_counts={s:sum(r["observation"]["street"]==s for r in records) for s in STREETS})
    return metadata,records


def save_corpus(path,metadata,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wt",encoding="utf-8") as stream:
        stream.write(json.dumps(dict(metadata=metadata))+"\n")
        for row in records:
            stream.write(json.dumps(row,separators=(",",":"))+"\n")


def load_corpus(path):
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        metadata=json.loads(next(stream))["metadata"]
        records=[json.loads(line) for line in stream]
    return metadata,records


def warm_replay(records,decisions=100000,warmup=2000,params=None):
    measured=Timing()
    _hero_equity.cache_clear()
    table=None
    bot=None
    started=perf_counter()
    cpu_started=process_time_ns()
    for i in range(decisions+warmup):
        row=records[i%len(records)]
        if row["table"]!=table or i%len(records)==0:
            table=row["table"]
            bot=Agent(seed=row["seed"]^120012,params=params)
        measured.act(bot,row["observation"],record=i>=warmup)
    return dict(seconds=perf_counter()-started,batch_process_cpu_ms=(process_time_ns()-cpu_started)/1e6,
                batch_cpu_note="Includes warmup, Agent construction and validation outside act timers.",warmup_decisions=warmup,
                initialization_timed=False,module_assets_already_imported=True,
                table_state_replayed_in_order_and_reset_at_boundaries=True,**measured.report())


def cold_replay(records,decisions=2000,seed=734000,params=None):
    measured=Timing()
    order=list(range(len(records)))
    random.Random(seed).shuffle(order)
    started=perf_counter()
    cpu_started=process_time_ns()
    for i in range(decisions):
        row=records[order[i%len(order)]]
        bot=Agent(seed=row["seed"]^120012,params=params)
        _hero_equity.cache_clear()  # Explicit module-cache cold miss, outside timer.
        measured.act(bot,row["observation"])
    return dict(seconds=perf_counter()-started,batch_process_cpu_ms=(process_time_ns()-cpu_started)/1e6,
                batch_cpu_note="Includes per-case Agent construction/cache clearing and validation outside act timers.",initialization_timed=False,
                module_assets_already_imported=True,fresh_Agent_each_decision=True,
                icm_cache_cleared_before_each_decision=True,
                note="First act on an in-hand observation with empty seat priors; process/asset startup is excluded.",
                **measured.report())


def _artificial_stress_obs(index,seed):
    """A legal 8-seat, 2000-total-chip state with six folders and HU all-in.

    Unequal initial stacks are intentional: this is a latency stress case,
    not one of the contest's equal-2000-chip starting tables.
    """
    rng=random.Random((seed<<24)^index)
    hero_start=210+index%157
    villain_start=220+(index*17)%149
    remaining=2000-hero_start-villain_start
    cuts=sorted(rng.sample(range(1,remaining-6*25),5))
    boundaries=[0,*cuts,remaining-6*25]
    other_stacks=[boundaries[i+1]-boundaries[i]+25 for i in range(6)]
    street=STREETS[index%4]
    nboard=(0,3,4,5)[index%4]
    known=rng.sample(range(52),2+nboard)
    history=[dict(seat=0,street="preflop",action="small_blind",amount=10,paid=10),
             dict(seat=1,street="preflop",action="big_blind",amount=20,paid=20)]
    history.extend(dict(seat=i,street="preflop",action="fold",amount=0,paid=0) for i in range(2,8))
    if nboard:
        history.extend([dict(seat=0,street="preflop",action="call",amount=20,paid=10),
                        dict(seat=1,street="preflop",action="check",amount=20,paid=0)])
        for prior in STREETS[1:index%4]:
            history.extend(dict(seat=i,street=prior,action="check",amount=0,paid=0) for i in (0,1))
        history.append(dict(seat=0,street=street,action="check",amount=0,paid=0))
        own_contribution=20
        villain_bet=villain_start-20
        street_bets=[0,villain_bet,*([0]*6)]
    else:
        # SB calls then BB raises all-in; all six intervening seats folded.
        history.append(dict(seat=0,street=street,action="call",amount=20,paid=10))
        own_contribution=20
        villain_bet=villain_start
        street_bets=[20,villain_bet,*([0]*6)]
    history.append(dict(seat=1,street=street,action="raise",amount=villain_bet,
                        paid=villain_start-20))
    stacks=[hero_start-own_contribution,0,*other_stacks]
    contributions=[own_contribution,villain_start,*([0]*6)]
    assert sum(stacks)+sum(contributions)==2000
    assert all(s>0 for s in stacks[2:])
    call=min(stacks[0],villain_bet-street_bets[0])
    return dict(hole=known[:2],board=known[2:],hero=0,button=7,street=street,
                street_index=index%4,stacks=stacks,contributions=contributions,
                street_contributions=street_bets,in_hand=[True,True,*([False]*6)],
                all_in=[False,True,*([False]*6)],pot=sum(contributions),
                current_bet=villain_bet,min_raise_to=2*villain_bet-(20 if not nboard else 0),
                max_raise_to=stacks[0]+street_bets[0],to_call=call,
                legal_actions=["fold","call"],small_blind=10,big_blind=20,
                hand_number=index,history=history,payouts=(.5,.3,.2,0,0,0,0,0),
                active_players=8,start_active=8,raise_reopened=False)


def stress_icm(decisions=2000,seed=736000,params=None):
    measured=Timing()
    bot=Agent(seed=seed,params=params)
    unique=set()
    zero_miss=0
    started=perf_counter()
    cpu_started=process_time_ns()
    for i in range(decisions):
        obs=_artificial_stress_obs(i,seed)
        unique.add(tuple(obs["stacks"])+tuple(obs["contributions"]))
        _hero_equity.cache_clear()
        measured.act(bot,obs)
        zero_miss+=_hero_equity.cache_info().misses==0
    return dict(seconds=perf_counter()-started,batch_process_cpu_ms=(process_time_ns()-cpu_started)/1e6,
                batch_cpu_note="Includes artificial state construction/cache clearing and validation outside act timers.",artificial=True,
                note="Eight live table stacks, six folded hands, heads-up all-in; unequal starting stacks total2000 chips. This measures an artificial worst-size ICM context, not game frequency.",
                unique_stack_states=len(unique),icm_cache_cleared_before_each_decision=True,
                decisions_without_icm_cache_miss=zero_miss,initialization_timed=False,
                **measured.report())


def deadline_trials(tables=256,seed_start=740000,params=None,max_hands=200):
    rows=[]
    started=perf_counter()
    for index in range(tables):
        seed=seed_start+index
        seat=balanced_hero_seat(index)
        scenario=("mixed","selfplay","shove")[index%3]
        row=dict(seed=seed,hero_seat=seat,scenario=scenario)
        for enforce in (False,True):
            bot=Agent(seed=seed^120012,params=params)
            opponents=make_lineup(scenario,seed=seed*17)
            random.Random(seed^0x12B0).shuffle(opponents)
            opponents.insert(seat,bot)
            result=run_tournament(opponents,seed,max_hands=max_hands,
                                  blind_schedule=SCHEDULES["doubling"],payouts=PAYOUTS["top3"],
                                  decision_limit_ns=1_000_000 if enforce else None)
            row["enforced" if enforce else "unbounded"]=dict(
                points=result["points"][seat],field_points=(sum(result["points"])-result["points"][seat])/7,
                decisions=result["decisions"][seat],latencies_ns=result["latencies_ns"][seat],
                invalid=result["invalid_actions"][seat],exceptions=result["exceptions"][seat],
                external_timeout_fallbacks=result["timeouts"][seat],internal_fallbacks=bot.fallbacks,
                table_timeout_fallbacks=sum(result["timeouts"]),capped=result["capped"])
        rows.append(row)
    aggregate={}
    for mode in ("unbounded","enforced"):
        cases=[r[mode] for r in rows]
        aggregate[mode]=dict(points=summary([r["points"] for r in cases]),
                             field_points=summary([r["field_points"] for r in cases]),
                             latency=_latencies([v for r in cases for v in r["latencies_ns"]]),
                             invalid_actions=sum(r["invalid"] for r in cases),
                             exceptions=sum(r["exceptions"] for r in cases),
                             external_timeout_fallbacks=sum(r["external_timeout_fallbacks"] for r in cases),
                             all_table_timeout_fallbacks=sum(r["table_timeout_fallbacks"] for r in cases),
                             internal_fallbacks=sum(r["internal_fallbacks"] for r in cases))
    for row in rows:
        for mode in ("unbounded","enforced"):
            row[mode].pop("latencies_ns")
    return dict(tables=tables,seed_start=seed_start,seconds=perf_counter()-started,
                synthetic_payouts=PAYOUTS["top3"],synthetic_blind_schedule=SCHEDULES["doubling"],
                same_policy_control=True,deadline_ns=1_000_000,deadline_applied_to_all_seats=True,
                metrics=aggregate,paired_points_enforced_minus_unbounded=summary(
                    [r["enforced"]["points"]-r["unbounded"]["points"] for r in rows]),
                table_rows=rows,
                note="Same-policy paired seeds; scheduling-driven timeouts can change all seats' play and later card assignment. This is local timing behavior, not an official performance estimate.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus",default="measurements/latency_corpus.jsonl.gz")
    parser.add_argument("--refresh-corpus",action="store_true")
    parser.add_argument("--corpus-tables",type=int,default=48)
    parser.add_argument("--seed-start",type=int,default=730000)
    parser.add_argument("--max-hands",type=int,default=200)
    parser.add_argument("--warm-decisions",type=int,default=100000)
    parser.add_argument("--warmup",type=int,default=2000)
    parser.add_argument("--cold-decisions",type=int,default=2000)
    parser.add_argument("--stress-decisions",type=int,default=2000)
    parser.add_argument("--enforce-deadline",action="store_true")
    parser.add_argument("--deadline-tables",type=int,default=256)
    parser.add_argument("--deadline-seed-start",type=int,default=740000)
    parser.add_argument("--params")
    parser.add_argument("--load-note",default="Other desktop/process load was not measured; no CPU/process/job inventory queried. Run serially after training/tuning for final evidence.")
    parser.add_argument("--output",default="measurements/latency.json")
    args=parser.parse_args()
    if not 1<=args.corpus_tables<=256 or not 1<=args.max_hands<=200:
        parser.error("corpus-tables in1..256 and max-hands in1..200 required")
    if any(not 0<=n<=1_000_000 for n in (args.warm_decisions,args.warmup,args.cold_decisions,args.stress_decisions)):
        parser.error("decision counts must be in0..1000000")
    if not 0<=args.deadline_tables<=2000:
        parser.error("deadline-tables must be in0..2000")
    params=None
    if args.params:
        value=json.loads(_local_path(args.params).read_text())
        params=value.get("params",value)
    started=perf_counter()
    corpus_path=_local_path(args.corpus)
    if args.refresh_corpus or not corpus_path.exists():
        metadata,records=collect_corpus(tables=args.corpus_tables,seed_start=args.seed_start,
                                        max_hands=args.max_hands,params=params)
        save_corpus(corpus_path,metadata,records)
    else:
        metadata,records=load_corpus(corpus_path)
    measured=dict(environment=dict(python=sys.version,platform=platform.platform(),
                                   machine=platform.machine(),processor=platform.processor(),
                                   gc_enabled=gc.isenabled(),load_note=args.load_note,
                                   timer="time.perf_counter_ns wall; time.process_time_ns CPU"),
                  clock_info={name:vars(time.get_clock_info(name)) for name in ("perf_counter","process_time")},
                  current_source_hashes=_hashes(),params=dict(DEFAULTS,**(params or {})),
                  corpus_path=str(corpus_path.relative_to(ROOT)),corpus=metadata,
                  corpus_source_matches_current=metadata["source_hashes"]==_hashes(),
                  warm=warm_replay(records,args.warm_decisions,args.warmup,params),
                  cold=cold_replay(records,args.cold_decisions,args.seed_start+4000,params),
                  artificial_uncached_icm=stress_icm(args.stress_decisions,args.seed_start+6000,params),
                  limitations=["Local Python/hardware timing; official hardware and adapter are unavailable.",
                               "Module import/model load, Agent construction, observation construction and JSON transport excluded.",
                               "Wall-clock scheduling pauses count as overruns; no hard deadline is guaranteed.",
                               "Replay uses recorded game states; artificial ICM stress is separately identified.",
                               "Fresh-Agent cold replay measures an empty table profile, not a fresh OS process."])
    if args.enforce_deadline and args.deadline_tables:
        measured["in_game_deadline"]=deadline_trials(args.deadline_tables,args.deadline_seed_start,params,args.max_hands)
    else:
        measured["in_game_deadline"]=dict(skipped=True,note="Pass --enforce-deadline to run a separately paired all-seat 1ms fallback trial.")
    measured["total_seconds"]=perf_counter()-started
    output=_local_path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(measured,indent=2))
    print(json.dumps(dict(output=str(output),corpus_observations=len(records),
                          warm=measured["warm"],cold=measured["cold"],
                          artificial_uncached_icm=measured["artificial_uncached_icm"],
                          in_game_deadline=(measured["in_game_deadline"].get("metrics") or measured["in_game_deadline"]),
                          total_seconds=measured["total_seconds"]),indent=2))


if __name__=="__main__":
    main()
