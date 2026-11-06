"""Exact Independent Chip Model utilities for explicitly supplied payout vectors.

Payouts are descending awards for the players STILL ALIVE, beginning with first
place. Drop tail places already awarded to eliminated players. Zero stacks get
zero future utility, not an inferred finishing-place award. Fewer payouts than
positive stacks means the remaining alive finishing places pay zero. Equal
stacks have equal expected utility; no arbitrary chip-count tie break is used.

The ICM assumes finish-order probabilities proportional to remaining stacks.
It is a scoring model, not a model of poker skill or future blind pressure.
No official contest payout structure is encoded here.
"""
from math import isfinite


def _validate(stacks, payouts):
    stacks = tuple(float(x) for x in stacks)
    payouts = tuple(float(x) for x in payouts)
    if any(not isfinite(x) or x < 0 for x in stacks):
        raise ValueError('stacks must be finite and nonnegative')
    if any(not isfinite(x) or x < 0 for x in payouts):
        raise ValueError('payouts must be finite and nonnegative')
    if any(a < b for a, b in zip(payouts, payouts[1:])):
        raise ValueError('payouts must be descending')
    active = [i for i, x in enumerate(stacks) if x > 0]
    if len(payouts) > len(active):
        raise ValueError('payout vector describes survivors only; remove awarded tail places')
    total = sum(stacks)
    if not isfinite(total):
        raise ValueError('total stack must be finite')
    return stacks, payouts, active, total


def _topology(n):
    size = 1 << n
    full = size - 1
    sums = [(mask, mask ^ (mask & -mask), (mask & -mask).bit_length() - 1)
            for mask in range(1, size)]
    nodes = [(mask, mask.bit_count(), full ^ mask,
              tuple((mask | (1 << j), j) for j in range(n) if not mask & (1 << j)))
             for mask in range(size)]
    return sums, nodes


# Construct fixed eight-seat combinatorics during import, outside decisions.
_TOPOLOGY = [_topology(n) for n in range(9)]


def _awards(payouts, n):
    # Everybody alive is guaranteed the lowest remaining award if all places
    # pay. Subtract it, then avoid exploring irrelevant zero-award tail places.
    base = payouts[-1] if len(payouts) == n else 0.0
    premiums = [p - base for p in payouts]
    while premiums and premiums[-1] == 0:
        premiums.pop()
    return base, premiums


def utility(stacks, payouts, hero):
    """Expected future payout, O(n*2**n); precomputed eight-seat topology.

    Hero indexes the original stacks including zero-stack eliminated seats.
    """
    stacks, payouts, active, total = _validate(stacks, payouts)
    if type(hero) is not int or not 0 <= hero < len(stacks):
        raise ValueError('hero index outside stacks')
    if not stacks[hero] or not payouts:
        return 0.0
    base, awards = _awards(payouts, len(active))
    if not awards:
        return base
    weights = [stacks[i] / total for i in active if i != hero]
    hero_weight = stacks[hero] / total
    n = len(weights)
    topology = _TOPOLOGY[n] if n < len(_TOPOLOGY) else _topology(n)
    subset = [0.0] * (1 << n)
    for mask, previous, j in topology[0]:
        subset[mask] = subset[previous] + weights[j]
    probabilities = [0.0] * len(subset)
    probabilities[0] = 1.0
    result = base
    depth = len(awards)
    for mask, rank, complement, transitions in topology[1]:
        if rank >= depth:
            continue
        # Complement sums stay positive even for extremely unequal stacks.
        scaled = probabilities[mask] / (hero_weight + subset[complement])
        result += scaled * hero_weight * awards[rank]
        if rank + 1 < depth:
            for target, j in transitions:
                probabilities[target] += scaled * weights[j]
    return result


def utilities(stacks, payouts):
    """Expected future payouts for all original seats, sharing one subset DP."""
    stacks, payouts, active, total = _validate(stacks, payouts)
    result = [0.0] * len(stacks)
    if not active or not payouts:
        return result
    n = len(active)
    base, awards = _awards(payouts, n)
    for i in active:
        result[i] = base
    if not awards:
        return result
    weights = [stacks[i] / total for i in active]
    topology = _TOPOLOGY[n] if n < len(_TOPOLOGY) else _topology(n)
    subset = [0.0] * (1 << n)
    for mask, previous, j in topology[0]:
        subset[mask] = subset[previous] + weights[j]
    probabilities = [0.0] * len(subset)
    probabilities[0] = 1.0
    depth = len(awards)
    for mask, rank, complement, transitions in topology[1]:
        if rank >= depth:
            continue
        scaled = probabilities[mask] / subset[complement]
        for target, j in transitions:
            probability = scaled * weights[j]
            result[active[j]] += probability * awards[rank]
            if rank + 1 < depth:
                probabilities[target] += probability
    return result
