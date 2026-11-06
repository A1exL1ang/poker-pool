"""Strict-timed tournaments for frozen twenty-bot pool and two Claude baselines.

No strategy changes. Adapter conversion and wrapper dispatch are inside timing.
Imports, construction, and resets are outside the decision clock, as before.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path
from schedule20 import BASELINES, PAYOUTS, VERSION, iter_matches, keyed_int, metadata, pilot_field_indices


def make_adapter(number):
    if number in BASELINES:
        from adapters import Baseline
        return Baseline(BASELINES[number])
    if number <= 8:
        from adapters import Adapter
    elif number <= 14:
        from adapters_9_14 import Adapter
    else:
        from adapters_15_20 import Adapter
    return Adapter(number)


class SeedSeparated:
    """Ignore the engine's deal-derived reset seed; never expose it to policies."""
    def __init__(self, adapter, number, cluster_id):
        self.adapter = adapter
        self.number = number
        self.policy_seed = keyed_int("policy", cluster_id, number)
        self.opaque_table_id = "table-" + format(keyed_int("opaque-id",cluster_id), "016x")

    def reset(self, ignored_deal_seed, seat, n_seats):
        self.adapter.reset(self.policy_seed, seat, n_seats)
        if hasattr(self.adapter, "table_id"):
            self.adapter.table_id = self.opaque_table_id

    def act(self, observation):
        return self.adapter.act(observation)

    def counters(self):
        a = self.adapter
        if self.number <= 8:
            # Original adapters do not consolidate policy.errors themselves.
            internal = getattr(a, "internal_errors", 0) + getattr(a.bot, "errors", 0)
        else:
            internal = getattr(a, "internal_errors", 0)
        return dict(internal_errors=int(internal),
                    safety_fallbacks=int(getattr(a,"safety_fallbacks",0)),
                    budget_fallbacks=int(getattr(a,"budget_fallbacks",0)),
                    native_fallbacks=int(getattr(a,"native_fallbacks",0)))


def audit_result(row):
    for key in ("points","final_stacks","placements","decisions","timeouts","errors","illegal"):
        if len(row[key]) != 8: raise ValueError(f"invalid result shape: {key}")
    if sum(row["final_stacks"]) != 16000: raise ValueError("chip conservation failed")
    if abs(sum(row["points"])-1) > 1e-9: raise ValueError("prize conservation failed")
    if len(set(row["seats"])) != 8: raise ValueError("duplicate entrant")
    for key in ("decisions","timeouts","errors","illegal","internal_errors","safety_fallbacks","budget_fallbacks","native_fallbacks"):
        if any(v<0 for v in row[key]): raise ValueError(f"negative counter: {key}")
    # Record errors and illegal actions for diagnosis; the final audit must reject
    # any comparison with unexplained faults rather than silently discard tables.


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=1)
    p.add_argument("--repetitions",type=int,default=4)
    p.add_argument("--cohort",choices=("all","pool","mixed"),default="all")
    p.add_argument("--pilot",action="store_true",help="one repetition, small all-entrant field cover, top3; not balanced production evidence")
    p.add_argument("--overwrite",action="store_true",help="refuse overwriting by default; explicitly clear this output first")
    a=p.parse_args()
    if a.shards<1 or not 0<=a.shard<a.shards: p.error("invalid shard")
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    meta=metadata(1 if a.pilot else a.repetitions)
    meta.update(cohort=a.cohort,pilot=a.pilot,shard=a.shard,shards=a.shards,
                engine_assumptions={"seats":8,"starting_stack":2000,"initial_blinds":[10,20],
                  "hands_per_level":25,"blind_multiplier":2,"max_hands":200,
                  "hand_cap":"ICM","timing_ns":1000000,"strict_timing":True,"official_compatibility":False})
    meta=json.loads(json.dumps(meta))  # canonicalize tuple/list and integer keys for resumability
    meta_path=out.with_suffix(out.suffix+".meta.json")
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError("existing output belongs to a different schedule/configuration")
    meta_path.write_text(json.dumps(meta,indent=2))
    done=set()
    if out.exists() and not a.overwrite:
        # Partial/truncated last rows deliberately fail closed; repair explicitly.
        for line in out.read_text().splitlines():
            row=json.loads(line)
            if row["match_id"] in done: raise ValueError("duplicate existing match ID")
            done.add(row["match_id"])
    from common.engine import Config, run_tournament
    written=0
    pilot_fields={c:set(pilot_field_indices(c)) for c in ("pool","mixed")} if a.pilot else {}
    with out.open("w" if a.overwrite else "a",buffering=1) as f:
        for index, match in enumerate(iter_matches(1 if a.pilot else a.repetitions,a.cohort,
                         ("top3",) if a.pilot else None)):
            if a.pilot and match["field_index"] not in pilot_fields[match["cohort"]]: continue
            if index%a.shards!=a.shard or match["match_id"] in done: continue
            bots=[SeedSeparated(make_adapter(n),n,match["cluster_id"]) for n in match["seats"]]
            start=time.perf_counter()
            r=run_tournament(bots,Config(payouts=PAYOUTS[match["payout"]],strict_timing=True),match["seed"])
            row=dict(match,version=VERSION,elapsed_s=time.perf_counter()-start,**dataclasses.asdict(r))
            row.pop("hand_records",None)
            counters=[b.counters() for b in bots]
            for name in counters[0]: row[name]=[c[name] for c in counters]
            audit_result(row)
            f.write(json.dumps(row,separators=(",",":"))+"\n")
            written+=1
    print(json.dumps(dict(complete=True,shard=a.shard,written=written,output=str(out))),flush=True)


if __name__ == "__main__":
    main()
