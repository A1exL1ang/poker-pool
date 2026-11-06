"""Exact balanced eight-seat schedules; no poker/model computation.

Fields are balanced *multisets*, deliberately repeating some compositions.
The affine group on F19 acts on finite points and fixes point19 (infinity).
Orbits are deduplicated individually, then repeated with explicit multiplicity.
Seed-cluster unit is (cohort, repetition, field_index), NOT repetition alone.
"""
from collections import Counter
from hashlib import sha256
from itertools import combinations
from random import Random

VERSION = "pool20_v1"
PAYOUTS = {
    "top3": (.5,.3,.2,0,0,0,0,0),
    "winner": (1,0,0,0,0,0,0,0),
    "top4": (.4,.3,.2,.1,0,0,0,0),
}
BASELINES = {21: "ppo", 22: "net"}

def keyed_int(domain, *parts):
    data = ":".join(map(str,(VERSION,domain,*parts))).encode()
    return int.from_bytes(sha256(data).digest()[:8], "big")

def affine_orbit(block):
    return sorted({tuple(sorted(19 if x == 19 else (a*x+b)%19 for x in block))
                   for a in range(1,19) for b in range(19)})

def design(cohort):
    h6 = sorted({pow(x,3,19) for x in range(1,19)})
    if cohort == "pool":
        fields = affine_orbit([19,0,*h6])*2 + affine_orbit([1,18,2,17,3,16,4,15])
        expected = (285,8,114,42)
    elif cohort == "mixed":
        fields = affine_orbit([19,0,1,18,2,17]) + affine_orbit(h6)*7
        expected = (570,6,171,45)
    else:
        raise ValueError(cohort)
    counts = Counter(x for field in fields for x in field)
    pairs = Counter(pair for field in fields for pair in combinations(field,2))
    b,k,r,lam = expected
    assert len(fields)==b and all(len(set(f))==k for f in fields)
    assert set(counts)==set(range(20)) and set(counts.values())=={r}
    assert len(pairs)==190 and set(pairs.values())=={lam}
    return fields

def fields_for(cohort, repetition):
    labels=list(range(1,21))
    Random(keyed_int("labels",cohort,repetition)).shuffle(labels)
    result=[]
    for index,field in enumerate(design(cohort)):
        seats=[labels[x] for x in field]
        if cohort=="mixed": seats.extend((21,22))
        # Rotation still gives every entrant every seat once per field.
        Random(keyed_int("field-order",cohort,repetition,index)).shuffle(seats)
        result.append(tuple(seats))
    return result

def iter_matches(repetitions=4, cohort="all", payouts=None, field_limit=None):
    if repetitions < 1: raise ValueError("repetitions must be positive")
    cohorts=("pool","mixed") if cohort=="all" else (cohort,)
    payouts=tuple(PAYOUTS) if payouts is None else tuple(payouts)
    for c in cohorts:
        for rep in range(repetitions):
            for field_index,field in enumerate(fields_for(c,rep)):
                if field_limit is not None and field_index>=field_limit: break
                cluster=f"{c}:{rep}:{field_index}"
                deck_seed=keyed_int("deck",c,rep,field_index)
                for payout in payouts:
                    if payout not in PAYOUTS: raise ValueError(payout)
                    for rotation in range(8):
                        seats=field[rotation:]+field[:rotation]
                        yield dict(match_id=f"{cluster}:{payout}:{rotation}",
                                   cluster_id=cluster,cohort=c,repetition=rep,
                                   field_index=field_index,payout=payout,
                                   rotation=rotation,seats=list(seats),seed=deck_seed)

def metadata(repetitions=4):
    result={"version":VERSION,"repetitions":repetitions,
            "tables":20520*repetitions,"payouts":PAYOUTS,"baselines":BASELINES,
            "cluster_unit":"cohort:repetition:field_index; all seats and payouts share deck seed",
            "policy_seeds":"domain-separated hashes of entrant and cluster; independent of deck seed",
            "fields_are_multisets":True,"cohorts":{}}
    for c in ("pool","mixed"):
        fields=design(c);k=len(fields[0]);r=len(fields)*k//20
        result["cohorts"][c]={"fields_per_repetition":len(fields),
            "unique_compositions_per_repetition":len(set(fields)),"bots_per_field":k,
            "fields_per_bot":r,"fields_per_bot_pair":r*(k-1)//19,
            "tables_per_bot_per_payout_per_repetition":r*8,
            "tables_per_bot_per_seat_per_payout_per_repetition":r}
    return result


def pilot_field_indices(cohort, repetition=0):
    """Small deterministic cover: every policy receives all eight seats."""
    fields=fields_for(cohort,repetition)
    unseen=set(range(1,21)); selected=[]
    while unseen:
        i=max((i for i in range(len(fields)) if i not in selected),
              key=lambda i:(len(unseen.intersection(fields[i])),-i))
        selected.append(i); unseen.difference_update(fields[i])
    return tuple(selected)
