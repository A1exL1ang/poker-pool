"""Exact ICM for local scoring; bounded approximation for gameplay risk.

No payout structure is assumed. Equal payouts or winner-take-all create no
risk premium. The gameplay approximation is deliberately conservative.
"""
from functools import lru_cache


def icm(stacks, payouts):
    stacks = tuple(float(s) for s in stacks)
    n = len(stacks)
    payouts = tuple(payouts[:n]) + (0.0,) * max(0, n - len(payouts))
    if not n:
        return ()

    @lru_cache(None)
    def recurse(mask):
        remaining = [i for i in range(n) if mask >> i & 1]
        rank = n - len(remaining)
        result = [0.] * n
        total = sum(stacks[i] for i in remaining)
        for i in remaining:
            probability = stacks[i] / total if total else 1. / len(remaining)
            result[i] += probability * payouts[rank]
            if len(remaining) > 1:
                sub = recurse(mask ^ (1 << i))
                for j in remaining:
                    result[j] += probability * sub[j]
        return tuple(result)

    return recurse((1 << n) - 1)


def risk_premium(players, hero, payouts, exposure):
    """Approximate equity premium for calling an elimination-threatening bet.

    Uses payout jumps, coverage and shorter stacks. This is not exact ICM.
    exposure is the fraction of the hero stack at risk, in [0,1].
    """
    n = len(players)
    if not payouts or n <= 2 or exposure <= 0:
        return 0.0
    pay = tuple(payouts[:n]) + (0.,) * max(0, n - len(payouts))
    span = pay[0] - pay[-1]
    if span <= 0:
        return 0.0
    survival = (pay[1] - pay[-1]) / span
    if survival <= 0:
        return 0.0
    hs = next(p.stack + p.contributed for p in players if p.seat == hero)
    shorter = sum(p.seat != hero and 0 < p.stack + p.contributed < hs * .55 for p in players)
    covering = sum(p.seat != hero and p.stack + p.contributed >= hs * .95 for p in players)
    near_jump = (pay[-2] - pay[-1]) / span
    premium = survival * (.025 + .012 * shorter) + near_jump * (.035 + .025 * shorter)
    if not covering:
        premium *= .35
    return min(.14, premium * exposure)
