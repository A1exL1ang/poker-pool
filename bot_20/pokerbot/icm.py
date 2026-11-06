"""Exact independent chip model for at most eight tournament seats.

Payouts are inputs, never a claim about the contest's unreleased payout schedule.
The implementation uses the usual sequential chip-proportional finish model.
"""

from math import isfinite


def icm_equities(stacks, payouts):
    """Return each seat's expected payout under the independent chip model.

    Missing trailing payouts mean zero. Zero stacks tie uniformly for the last
    places; callers evaluating a live tournament normally pass survivors only.
    This is exact enumeration by subset dynamic programming, O(n * 2**n).
    """
    n = len(stacks)
    if n > 8:
        raise ValueError("at most eight stacks are supported")
    if not all(isfinite(x) and x >= 0 for x in stacks):
        raise ValueError("stacks must be finite and nonnegative")
    if not all(isfinite(x) and x >= 0 for x in payouts):
        raise ValueError("payouts must be finite and nonnegative")
    if len(payouts) > n:
        raise ValueError("pass only payouts for the supplied seats")
    if not n:
        return []
    awards = list(payouts) + [0.0] * (n - len(payouts))
    active = [i for i, stack in enumerate(stacks) if stack > 0]
    result = [0.0] * n
    m = len(active)
    if m < n:
        residual = sum(awards[m:]) / (n - m)
        for i, stack in enumerate(stacks):
            if stack == 0:
                result[i] = residual
    if not m:
        return result
    chips = [float(stacks[i]) for i in active]
    total = sum(chips)
    paid_ranks = [rank for rank in range(m) if awards[rank] != 0]
    if not paid_ranks:
        return result
    last_paid = paid_ranks[-1]
    size = 1 << m
    mass = [0.0] * size
    removed = [0.0] * size
    mass[0] = 1.0
    for mask in range(size - 1):
        if mask:
            bit = mask & -mask
            removed[mask] = removed[mask ^ bit] + chips[bit.bit_length() - 1]
        denominator = total - removed[mask]
        rank = mask.bit_count()
        if rank > last_paid:
            continue
        # A single remaining seat avoids loss of precision from subtraction.
        if rank == m - 1:
            j = ((size - 1) ^ mask).bit_length() - 1
            result[active[j]] += mass[mask] * awards[rank]
            continue
        scale = mass[mask] / denominator
        remaining = (size - 1) ^ mask
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length() - 1
            probability = scale * chips[j]
            mass[mask | bit] += probability
            result[active[j]] += probability * awards[rank]
            remaining ^= bit
    return result


def icm_hero(stacks, payouts, hero):
    """Exact one-seat ICM, enumerating only subsets that exclude the hero."""
    if not 0 <= hero < len(stacks):
        raise ValueError("hero index outside stacks")
    n = len(stacks)
    if n > 8 or len(payouts) > n:
        raise ValueError("at most eight seats; payouts must match supplied seats")
    if not all(isfinite(x) and x >= 0 for x in stacks):
        raise ValueError("stacks must be finite and nonnegative")
    if not all(isfinite(x) and x >= 0 for x in payouts):
        raise ValueError("payouts must be finite and nonnegative")
    awards = list(payouts) + [0.0] * (n - len(payouts))
    active = [i for i, stack in enumerate(stacks) if stack > 0]
    if stacks[hero] == 0:
        return sum(awards[len(active):]) / (n - len(active))
    opponents = [float(stacks[i]) for i in active if i != hero]
    own = float(stacks[hero])
    total = own + sum(opponents)
    size = 1 << len(opponents)
    mass = [0.0] * size
    removed = [0.0] * size
    mass[0] = 1.0
    result = 0.0
    paid_ranks = [rank for rank in range(len(active)) if awards[rank] != 0]
    if not paid_ranks:
        return result
    last_paid = paid_ranks[-1]
    for mask in range(size):
        if mask:
            bit = mask & -mask
            removed[mask] = removed[mask ^ bit] + opponents[bit.bit_length() - 1]
        rank = mask.bit_count()
        if rank > last_paid:
            continue
        if mask == size - 1:
            result += mass[mask] * awards[rank]
            continue
        scale = mass[mask] / (total - removed[mask])
        result += scale * own * awards[rank]
        if rank == last_paid:
            continue
        remaining = (size - 1) ^ mask
        while remaining:
            bit = remaining & -remaining
            mass[mask | bit] += scale * opponents[bit.bit_length() - 1]
            remaining ^= bit
    return result


# Short alias useful for notebook experiments and independent validation.
icm = icm_equities
