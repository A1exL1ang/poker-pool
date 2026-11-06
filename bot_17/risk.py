"""Exact small-table ICM. Only caller-supplied payouts are used."""


def icm_equity(stacks, payouts, hero=0):
    """Expected points under independent exponential chip races, subset DP.

    Only positive stacks participate. Eliminated places must have been paid by
    the tournament engine already. Complexity O(n 2**(n-1)), n <= 8.
    """
    if stacks[hero] <= 0:
        return float(payouts[len(stacks) - 1]) if len(payouts) >= len(stacks) else 0.0
    live = [float(x) for x in stacks if x > 0]
    hs = float(stacks[hero])
    others = [float(s) for i, s in enumerate(stacks) if i != hero and s > 0]
    n = len(others)
    pay = list(payouts[: n + 1]) + [0.0] * max(0, n + 1 - len(payouts))
    if not n:
        return float(pay[0])
    # Constant payouts and winner-take-all have exact cheap shortcuts.
    if all(x == pay[-1] for x in pay[1:]):
        return pay[-1] + (pay[0] - pay[-1]) * hs / sum(live)
    size = 1 << n
    mass = [0.0] * size
    removed = [0.0] * size
    mass[0] = 1.0
    total = sum(live)
    baseline = pay[-1]
    last_varying = max(i for i, value in enumerate(pay) if value != baseline)
    result = float(baseline)
    for mask in range(size):
        if mask:
            bit = mask & -mask
            removed[mask] = removed[mask ^ bit] + others[bit.bit_length() - 1]
        place = mask.bit_count()
        if place > last_varying:
            continue
        denom = total - removed[mask]
        prob = mass[mask]
        result += prob * hs / denom * (pay[place] - baseline)
        if place == last_varying:
            continue
        remaining = (size - 1) ^ mask
        while remaining:
            bit = remaining & -remaining
            mass[mask | bit] += prob * others[bit.bit_length() - 1] / denom
            remaining ^= bit
    return result


def call_threshold(players, hero_seat, opponent_seat, call, eligible_pot, payouts):
    """Heads-up contested-pot ICM break-even win probability.

    Approximation when other players are still contesting or have unequal side
    pots; strategy invokes only when a single opponent remains. Folded chips
    already in the pot are restored to the winner. Chips committed by the hero
    are sunk. No imaginary payout schedule is supplied.
    """
    ordered = [p for p in players if p['stack'] + p['committed'] > 0]
    h = next(i for i, p in enumerate(ordered) if p['seat'] == hero_seat)
    v = next(i for i, p in enumerate(ordered) if p['seat'] == opponent_seat)
    # All chips in pot belong to the eventual winner in the heads-up case.
    pot = sum(p['committed'] for p in players)
    base = [p['stack'] for p in ordered]
    fold = base[:]
    fold[v] += pot
    win = base[:]
    win[h] += eligible_pot  # eligible_pot excludes the new call
    win[v] += max(0, pot - eligible_pot)
    lose = base[:]
    lose[h] -= call
    lose[v] += pot + call
    # Preserve zero stacks as finishing-place payouts in losing state.
    utility_fold = icm_equity(fold, payouts, h)
    utility_win = icm_equity(win, payouts, h)
    utility_lose = icm_equity(lose, payouts, h)
    span = utility_win - utility_lose
    return max(0.0, min(1.0, (utility_fold - utility_lose) / span)) if span > 1e-12 else 1.0
