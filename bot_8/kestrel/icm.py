"""Exact ICM for local scoring; optional risk adjustment in the policy.

Payouts are caller-supplied remaining finishing-place POINTS, not cash prizes.
Missing payouts mean chip EV. No official point schedule is assumed.
"""

_MASKS = {}
for _n in range(8):
    _MASKS[_n] = tuple(tuple(i for i in range(_n) if m & (1 << i)) for m in range(1 << _n))


def value(stacks, hero, payouts):
    if not payouts:
        return stacks[hero] / sum(stacks) if sum(stacks) else 0.0
    if stacks[hero] <= 0:
        return payouts[len(stacks) - 1]
    positive = [i for i, s in enumerate(stacks) if s > 0]
    pay = payouts[:len(positive)]
    base = pay[-1]
    if pay[0] == base:
        return float(base)
    h = float(stacks[hero])
    other = tuple(float(stacks[i]) for i in positive if i != hero)
    total = h + sum(other)
    masks = _MASKS[len(other)]
    size = len(masks)
    sums = [0.0] * size
    prob = [0.0] * size
    prob[0] = 1.0
    result = base
    for mask, members in enumerate(masks):
        rank = len(members)
        if pay[rank] == base:
            continue
        if mask:
            bit = mask & -mask
            idx = bit.bit_length() - 1
            sums[mask] = sums[mask ^ bit] + other[idx]
            p = 0.0
            for j in members:
                prev = mask ^ (1 << j)
                p += prob[prev] * other[j] / (total - sums[prev])
            prob[mask] = p
        result += prob[mask] * h / (total - sums[mask]) * (pay[rank] - base)
    return result


def call_threshold(obs, call, villain_seat):
    """Exact heads-up terminal-call ICM threshold under whole-pot eligibility.

    This is called only for expensive calls with one live opponent and no
    unequal all-in contribution. It models fold/win/lose right now, not future
    play or unobserved future hands. Ties use the equity approximation.
    """
    players = obs.players
    h = next(i for i, p in enumerate(players) if p.seat == obs.hero)
    v = next(i for i, p in enumerate(players) if p.seat == villain_seat)
    stacks = [p.stack for p in players]
    folded = stacks.copy()
    folded[v] += obs.pot
    won = stacks.copy()
    won[h] += obs.pot
    lost = stacks.copy()
    lost[h] -= call
    lost[v] += obs.pot + call
    u_fold = value(folded, h, obs.payouts)
    u_win = value(won, h, obs.payouts)
    u_lose = value(lost, h, obs.payouts)
    if u_win <= u_lose + 1e-12:
        return call / max(1, obs.pot + call)
    return max(0.0, min(1.0, (u_fold - u_lose) / (u_win - u_lose)))
