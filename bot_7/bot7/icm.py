"""Exact independent-chip-model utility for up to eight players."""
from functools import lru_cache
from math import comb


SUBSETS = tuple(tuple((mask ^ (mask & -mask), (mask & -mask).bit_length() - 1, mask.bit_count())
                      for mask in range(1, 1 << n)) for n in range(8))


@lru_cache(maxsize=128)
def coefficients(schedule):
    n = len(schedule) - 1
    return tuple(sum((-1 if r % 2 else 1) * comb(m, r) * schedule[n - m + r]
                     for r in range(m + 1)) for m in range(n + 1))


def utility(stacks, hero, payouts):
    own = stacks[hero]
    if own <= 0:
        return 0.0
    others = [x for i, x in enumerate(stacks) if i != hero and x > 0]
    n = len(others)
    if not n:
        return float(payouts[0])
    total = own + sum(others)
    relevant = min(n + 1, len(payouts))
    while relevant and payouts[relevant - 1] == 0:
        relevant -= 1
    if relevant == 1:
        return own / total * payouts[0]
    # Linear finishing points have an exact pairwise expression for expected
    # finishing position, avoiding exponential work for this common test case.
    schedule = payouts[:n + 1]
    if len(schedule) == n + 1:
        step = schedule[0] - schedule[1]
        if all(abs((a - b) - step) < 1e-12 for a, b in zip(schedule, schedule[1:])):
            return schedule[-1] + step * sum(own / (own + x) for x in others)
    # ICM is an exponential race with rates equal to chip stacks. Expanding
    # prod_j[z + (1-z)*exp(-stack_j*t)] and integrating against hero's
    # exponential density gives one term per opponent subset. The coefficient
    # depends only on its size and finite differences of the payout schedule:
    # utility = sum_T coefficient[|T|] * own / (own + sum(stacks in T)).
    # This is exact, with O(2**(n-1)) work and no permutation/transition loops.
    schedule = tuple(payouts[:n + 1]) + (0,) * max(0, n + 1 - len(payouts))
    terms = coefficients(schedule)
    subtotal = [0] * (1 << n)
    value = float(terms[0])
    for mask, (previous, index, count) in enumerate(SUBSETS[n], 1):
        amount = subtotal[previous] + others[index]
        subtotal[mask] = amount
        coefficient = terms[count]
        if coefficient:
            value += coefficient * own / (own + amount)
    return max(0.0, min(float(schedule[0]), value))


def call_threshold(stacks, hero, villain, pot, cost, payouts):
    """Heads-up pot, no side pots: assume villain wins immediately if we fold.

    Inputs are chips behind, so pot is added to the eventual winner. A call
    costs `cost`; all opposing wagers are already included in `pot`.
    """
    if cost <= 0:
        return 0.0
    base = list(stacks)
    folded = base.copy()
    folded[villain] += pot
    won = base.copy()
    won[hero] += pot
    lost = base.copy()
    lost[hero] -= cost
    lost[villain] += pot + cost
    # Zero stacks in loss receive the next elimination place, not zero, when
    # the point schedule pays every finishing position.
    def u(values):
        if values[hero] <= 0:
            survivors = sum(x > 0 for x in values)
            return payouts[survivors] if survivors < len(payouts) else 0.0
        return utility(values, hero, payouts)
    f, w, l = u(folded), u(won), u(lost)
    return max(0.0, min(1.0, (f - l) / (w - l))) if w > l else 1.0

