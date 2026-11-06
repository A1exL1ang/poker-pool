"""Exact single-player ICM, invoked only when real payouts are supplied."""
import ctypes
from .native import LIB


def hero_icm(stacks, hero, payouts):
    if LIB is not None and len(stacks) <= 8:
        n=len(stacks)
        return LIB.bot_icm((ctypes.c_double*n)(*stacks),n,hero,(ctypes.c_double*n)(*payouts[:n]))
    return hero_icm_python(stacks,hero,payouts)


def hero_icm_python(stacks, hero, payouts):
    n = len(stacks)
    if n == 1:
        return payouts[0]
    if stacks[hero] <= 0:
        return payouts[n-1]
    other = [s for i, s in enumerate(stacks) if i != hero and s > 0]
    h = stacks[hero]
    total = h + sum(other)
    size = 1 << len(other)
    sums = [0.] * size
    counts = [0] * size
    dp = [0.] * size
    dp[0] = 1.
    utility = 0.
    # Remove the common last-place payout. Only the ranks with a positive
    # increment matter; e.g. top-three payouts need subsets of size at most 2.
    base = payouts[min(n-1, len(payouts)-1)]
    last_rank = max((i for i in range(min(n, len(payouts))) if payouts[i] != base), default=-1)
    if last_rank < 0:
        return base
    utility = base
    for mask in range(size):
        if mask:
            bit = mask & -mask
            prev = mask ^ bit
            sums[mask] = sums[prev] + other[bit.bit_length()-1]
            counts[mask] = counts[prev] + 1
        rank = counts[mask]
        if rank > last_rank:
            continue
        remaining = total - sums[mask]
        p = dp[mask]
        utility += p * h / remaining * (payouts[rank] - base)
        if rank == last_rank:
            continue
        for j, s in enumerate(other):
            b = 1 << j
            if not mask & b:
                dp[mask | b] += p * s / remaining
    return utility


def call_threshold(stacks, hero, villain, pot, cost, payouts):
    """Terminal heads-up contested pot; other players may remain at the table.

    Stacks exclude all chips in pot; pot includes every prior contribution.
    Assumes no side pot and villain's wager already in pot. Splits are modeled
    by chip-equity interpolation, an acknowledged approximation under ICM.
    """
    fold = list(stacks)
    fold[villain] += pot
    win = list(stacks)
    win[hero] += pot
    lose = list(stacks)
    lose[hero] -= cost
    lose[villain] += pot + cost
    uf = hero_icm(fold, hero, payouts)
    uw = hero_icm(win, hero, payouts)
    ul = hero_icm(lose, hero, payouts)
    return max(0., min(1., (uf - ul) / max(1e-12, uw - ul)))
