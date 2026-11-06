"""Exact hero ICM equity, using a subset dynamic program (up to eight seats)."""
from functools import lru_cache


@lru_cache(maxsize=1024)
def icm_equity(stacks, hero, payouts):
    if len(stacks) > 8:
        raise ValueError("ICM calculator supports at most eight remaining players")
    if any(s < 0 for s in stacks) or any(p < 0 for p in payouts):
        raise ValueError("stacks and payouts must be nonnegative")
    if any(a < b for a, b in zip(payouts, payouts[1:])):
        raise ValueError("payouts must be ordered first to last")
    if stacks[hero] <= 0:
        return payouts[len(stacks) - 1] if len(payouts) >= len(stacks) else 0.0
    others = tuple(s for i, s in enumerate(stacks) if i != hero and s > 0)
    hero_stack = stacks[hero]
    total = hero_stack + sum(others)
    guaranteed = payouts[len(others)] if len(payouts) > len(others) else 0.0
    last_paid = min(len(others), len(payouts) - 1)
    while last_paid >= 0 and payouts[last_paid] == guaranteed:
        last_paid -= 1
    if last_paid < 0:
        return float(guaranteed)
    if last_paid == 0:
        return guaranteed + hero_stack / total * (payouts[0] - guaranteed)
    size = 1 << len(others)
    sums = [0.0] * size
    mass = [0.0] * size
    mass[0] = 1.0
    result = float(guaranteed)
    for mask in range(size):
        if mask:
            low = mask & -mask
            sums[mask] = sums[mask ^ low] + others[low.bit_length() - 1]
        remaining = total - sums[mask]
        rank = mask.bit_count()
        if rank > last_paid or mass[mask] == 0:
            continue
        payout = payouts[rank] - guaranteed if rank < len(payouts) else 0.0
        result += mass[mask] * hero_stack / remaining * payout
        if rank == last_paid:
            continue
        available = (size - 1) ^ mask
        while available:
            low = available & -available
            j = low.bit_length() - 1
            mass[mask | low] += mass[mask] * others[j] / remaining
            available ^= low
    return result


def risk_premium(stacks, hero, villain, payouts):
    """Pairwise finite-loss ICM premium over a 50% even-money gamble.

    This is a risk heuristic, not an exact multiway/all-in ICM action solver.
    It is zero for winner-take-all payouts and two-player play.
    """
    if len(stacks) < 3 or len(payouts) < 2 or payouts[1] <= 0:
        return 0.0
    transfer = min(stacks[hero], stacks[villain]) * 0.75
    if transfer <= 0:
        return 0.0
    base = icm_equity(tuple(stacks), hero, tuple(payouts))
    win = list(stacks)
    lose = list(stacks)
    win[hero] += transfer
    win[villain] -= transfer
    lose[hero] -= transfer
    lose[villain] += transfer
    gain = icm_equity(tuple(win), hero, tuple(payouts)) - base
    loss = base - icm_equity(tuple(lose), hero, tuple(payouts))
    if gain + loss <= 1e-12:
        return 0.0
    return max(0.0, min(0.20, loss / (gain + loss) - 0.5))

