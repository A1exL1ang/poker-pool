"""Exact small-table ICM without assuming a contest payout distribution."""

import math


def validate_payouts(payouts, n):
    if payouts is None:
        return None
    values = tuple(float(x) for x in payouts)
    if len(values) < n or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("payouts must give nonnegative points for every remaining place")
    if any(values[i] < values[i + 1] for i in range(len(values) - 1)):
        raise ValueError("payouts must be descending")
    return values[:n]


def equity(stacks, payouts, hero=0):
    """Malmuth-Harville ICM expectation, O(n*2**n), n <= 8.

    Zero-stack players occupy the last places; tied zero stacks split those
    places. Callers modeling simultaneous busts should supply engine results.
    """
    n = len(stacks)
    p = validate_payouts(payouts, n)
    if p is None:
        return float(stacks[hero])
    if not 0 <= hero < n or any(s < 0 for s in stacks):
        raise ValueError("invalid stacks/hero")
    positive = [i for i, s in enumerate(stacks) if s > 0]
    if hero not in positive:
        return sum(p[len(positive):]) / (n - len(positive))
    own = float(stacks[hero])
    others = [float(stacks[i]) for i in positive if i != hero]
    total = own + sum(others)
    size = 1 << len(others)
    probs = [0.0] * size
    sums = [0.0] * size
    probs[0] = 1.0
    out = 0.0
    for mask in range(size):
        if mask:
            bit = mask & -mask
            sums[mask] = sums[mask ^ bit] + others[bit.bit_length() - 1]
        remain = total - sums[mask]
        pr = probs[mask]
        out += pr * own / remain * p[mask.bit_count()]
        todo = size - 1 - mask
        while todo:
            bit = todo & -todo
            j = bit.bit_length() - 1
            probs[mask | bit] += pr * others[j] / remain
            todo ^= bit
    return out


def call_threshold(stacks_fold, stacks_win, stacks_lose, payouts, hero,
                   chip_threshold):
    """Binary win/lose break-even probability for supplied counterfactual stacks.

    This is exact ICM for the supplied outcomes, not an equilibrium strategy.
    Split pots and multiway outcomes require separate outcome modeling.
    """
    if payouts is None:
        return chip_threshold
    a = equity(stacks_fold, payouts, hero)
    b = equity(stacks_win, payouts, hero)
    c = equity(stacks_lose, payouts, hero)
    return max(0.0, min(1.0, (a - c) / (b - c))) if b > c else chip_threshold
