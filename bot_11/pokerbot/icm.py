"""Exact hero ICM utility for explicit, externally supplied finish payouts."""

from functools import lru_cache
import math


# For every opponent subset cache its predecessor, newly added opponent and
# cardinality.  The plans contain only 247 entries over all table sizes <=8.
_SUBSET_PLANS = tuple(
    tuple((mask ^ (mask & -mask), (mask & -mask).bit_length() - 1, mask.bit_count())
          for mask in range(1, 1 << size))
    for size in range(8)
)


@lru_cache(maxsize=64)
def _coefficients(payouts):
    if not 1 <= len(payouts) <= 8:
        raise ValueError('ICM supports one through eight remaining players')
    if any(not math.isfinite(p) for p in payouts):
        raise ValueError('ICM payouts must be finite')
    if any(a < b for a, b in zip(payouts, payouts[1:])):
        raise ValueError('ICM payouts must be nonincreasing by finishing place')
    # Coefficient k is the kth finite difference of the reversed payouts.
    # Conditional on hero's exponential arrival time t, each opponent arrives
    # first independently with probability 1-exp(-stack*t).  Expanding the
    # rank generating polynomial and integrating yields one term per subset A:
    # coefficient[|A|] * hero_stack / (hero_stack + sum(stacks in A)).
    row = list(reversed(payouts))
    coefficients = []
    while row:
        coefficients.append(row[0])
        row = [row[i + 1] - row[i] for i in range(len(row) - 1)]
    # A common baseline plus winner-take-all increment is exactly chip-linear.
    linear = all(value == 0 for value in coefficients[1:-1])
    return tuple(coefficients), linear


def utility(stacks, hero, payouts):
    """Expected payout under chip-proportional finish ordering (<=8 seats).

    Zero-stack players must have been removed before calling. No payout vector
    is implied by this module; contest prize money is never used as table points.
    """
    if len(stacks) != len(payouts) or not 0 <= hero < len(stacks):
        raise ValueError('ICM requires one explicit payout per remaining player')
    if any(not math.isfinite(s) or s <= 0 for s in stacks):
        raise ValueError('ICM requires finite positive stacks')
    coefficients, linear = _coefficients(tuple(payouts))
    if len(stacks) == 1:
        return payouts[0]
    others = [s for i, s in enumerate(stacks) if i != hero]
    n = len(others)
    mine = stacks[hero]
    if linear:
        return coefficients[0] + coefficients[-1] * mine / sum(stacks)
    sums = [0] * (1 << n)
    terms = [coefficients[0]]
    for mask, (prev, index, count) in enumerate(_SUBSET_PLANS[n], 1):
        subset_sum = sums[prev] + others[index]
        sums[mask] = subset_sum
        coefficient = coefficients[count]
        if coefficient:
            terms.append(coefficient * mine / (mine + subset_sum))
    # Finite differences may alternate signs. Compensated summation limits
    # cancellation while preserving the exact finite-subset formula.
    return max(payouts[-1], min(payouts[0], math.fsum(terms)))


def call_threshold(stacks, hero, villain, pot, cost, payouts):
    """ICM break-even showdown equity in an assumed heads-up pot resolution.

    stacks are uncommitted chips. Existing pot is awarded to villain on fold
    or loss, and to hero on win. Outstanding other calls/side pots are excluded;
    caller uses this only for a single live opponent and a large call.
    """
    if (not 0 <= hero < len(stacks) or not 0 <= villain < len(stacks) or hero == villain
            or any(not math.isfinite(s) or s < 0 for s in stacks)
            or not math.isfinite(pot) or pot < 0
            or not math.isfinite(cost) or not 0 <= cost <= stacks[hero]):
        raise ValueError('invalid heads-up ICM resolution')
    live = [i for i, s in enumerate(stacks) if s > 0 or i in (hero, villain)]
    # Payouts refer to original finishing places; currently eliminated seats
    # occupy the bottom places, so surviving seats use the leading entries.
    pay = list(payouts[:len(live)])
    if len(pay) != len(live) or len(live) > 8:
        raise ValueError('invalid payouts')
    _coefficients(tuple(pay))
    h, v = live.index(hero), live.index(villain)
    start = [stacks[i] for i in live]
    def value(chips):
        if chips[h] <= 0:
            return pay[-1]
        kept = [j for j, s in enumerate(chips) if s > 0]
        return utility([chips[j] for j in kept], kept.index(h), pay[:len(kept)])
    fold = start[:]
    fold[v] += pot
    lose = start[:]
    lose[h] -= cost
    lose[v] += pot + cost
    win = start[:]
    win[h] -= cost
    win[h] += pot + cost
    uf, ul, uw = value(fold), value(lose), value(win)
    return max(0.0, min(1.0, (uf - ul) / (uw - ul))) if uw > ul else 1.0

