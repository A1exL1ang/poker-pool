"""Exact Independent Chip Model for small tournament tables.

The finish-order model chooses each next finisher in proportion to remaining
chips. The subset dynamic program is exact for this model, not a prediction of
skill-adjusted tournament outcomes. Payouts are supplied explicitly by caller.
"""
import math


def icm(stacks, payouts):
    n = len(stacks)
    if len(payouts) != n or n > 16:
        raise ValueError("one payout per player; at most 16 players")
    if any(s < 0 or not math.isfinite(s) for s in stacks):
        raise ValueError("nonnegative finite stacks required")
    if any(not math.isfinite(p) for p in payouts):
        raise ValueError("finite payouts required")
    if n == 0:
        return []
    if sum(stacks) == 0:
        return [sum(payouts) / n] * n
    size = 1 << n
    mass = [0.0] * size
    mass[0] = 1.0
    sums = [0.0] * size
    for mask in range(1, size):
        bit = mask & -mask
        sums[mask] = sums[mask ^ bit] + stacks[bit.bit_length() - 1]
    total = sums[-1]
    result = [0.0] * n
    for mask in range(size - 1):
        rank = mask.bit_count()
        probability = mass[mask]
        remaining = total - sums[mask]
        for i, stack in enumerate(stacks):
            bit = 1 << i
            if not mask & bit:
                chance = stack / remaining if remaining > 0 else 1 / (n - rank)
                path = probability * chance
                mass[mask | bit] += path
                result[i] += path * payouts[rank]
    return result

