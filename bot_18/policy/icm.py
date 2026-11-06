"""Exact single-player ICM by subset dynamic programming.

Payouts are caller-supplied finishing points, highest place first. Nothing in
this module supplies or assumes the unknown contest payout distribution.
"""
from .icm_fast import fast_value


def _structure(m):
    out = []
    for mask in range(1 << m):
        parents = []
        for j in range(m):
            bit = 1 << j
            if mask & bit:
                parents.append((mask ^ bit, j))
        out.append((mask.bit_count(), parents))
    return out


_STRUCTURES = [_structure(m) for m in range(8)]


def value_reference(stacks, hero, payouts):
    """Expected points for hero; zero stacks are already eliminated.

    A newly eliminated hero receives the lowest place among this input group.
    Tied simultaneous eliminations must be handled by the caller.
    Complexity O(n 2**n), n <= 8 in the local contract.
    """
    n = len(stacks)
    if not n or len(payouts) < n:
        raise ValueError("one payout per represented player is required")
    hs = stacks[hero]
    if hs <= 0:
        return float(payouts[n - 1])
    others = [s for i, s in enumerate(stacks) if i != hero and s > 0]
    m = len(others)
    if not m:
        return float(payouts[0])
    # Constant payouts and winner-takes-all have particularly cheap solutions.
    base = payouts[m]
    if all(p == base for p in payouts[1:m + 1]):
        return base + (payouts[0] - base) * hs / (hs + sum(others))
    total = hs + sum(others)
    last_relevant = max((i for i, p in enumerate(payouts[:m + 1]) if p != base), default=-1)
    size = 1 << m
    sums = [0.0] * size
    prob = [0.0] * size
    prob[0] = 1.0
    result = float(base)
    for mask, (rank, parents) in enumerate(_STRUCTURES[m]):
        if rank > last_relevant:
            continue
        if mask:
            prior, j = parents[0]
            sums[mask] = sums[prior] + others[j]
            p = 0.0
            for prior, j in parents:
                p += prob[prior] * others[j] / (total - sums[prior])
            prob[mask] = p
        result += prob[mask] * hs / (total - sums[mask]) * (payouts[rank] - base)
    return result


def value(stacks, hero, payouts):
    """Exact ICM via compiled subset moments; readable DP remains above.

    The two forms agree to floating-point roundoff. Fixed n<=8 allows all
    subset arithmetic to be generated offline, keeping live decisions fast.
    """
    n = len(stacks)
    if not n or n > 8 or len(payouts) < n:
        raise ValueError('one payout per player, at most eight players required')
    hs = stacks[hero]
    if hs <= 0:
        return float(payouts[n - 1])
    others = tuple(s for i, s in enumerate(stacks) if i != hero and s > 0)
    m = len(others)
    if not m:
        return float(payouts[0])
    base = payouts[m]
    if all(p == base for p in payouts[1:m + 1]):
        return base + (payouts[0] - base) * hs / (hs + sum(others))
    differences = tuple(p - base for p in payouts[:m])
    last_relevant = max(i for i, p in enumerate(differences) if p != 0)
    return base + fast_value(hs, others, differences, last_relevant)


def call_threshold(players, hero_seat, villain_seat, pot, call, payouts):
    """Heads-up pot's ICM break-even call equity.

    All invested chips are in pot, remaining stacks exclude bets. Use only
    when exactly two players contest the pot, no side-pot cap is involved,
    and the call closes betting (or as a stated terminal approximation).
    Fold awards current pot to villain; win awards pot+call to hero; loss
    awards pot+call to villain. This conserves total chips in each branch.
    """
    alive = [p for p in players if p['stack'] + p['invested'] > 0]
    seats = [p['seat'] for p in alive]
    if hero_seat not in seats or villain_seat not in seats or len(payouts) < len(alive):
        return call / max(1, pot + call)
    h, v = seats.index(hero_seat), seats.index(villain_seat)
    fold = [p['stack'] for p in alive]
    fold[v] += pot
    win = fold.copy()
    win[v] -= pot
    win[h] += pot
    lose = fold.copy()
    lose[h] -= call
    lose[v] += call
    ef = value(fold, h, payouts)
    ew = value(win, h, payouts)
    el = value(lose, h, payouts)
    if ew <= el + 1e-15:
        return 1.0
    return min(1.0, max(0.0, (ef - el) / (ew - el)))


def shove_gain(players, hero_seat, pot, payouts, fold_probability, callers):
    """Approximate open-shove tournament utility, exact ICM at each leaf.

    callers is (seat, first-caller probability, showdown equity) tuples.
    All remaining opponents must have chips; at most one is modeled calling.
    Uncalled excess is returned in each leaf. Fold awards current pot to the
    largest existing bettor (a terminal baseline, not a future-game solver).
    Caller weights plus fold_probability should total one.
    """
    alive = [p for p in players if p['stack'] + p['invested'] > 0]
    seats = [p['seat'] for p in alive]
    h = seats.index(hero_seat)
    me = alive[h]
    current = [p['stack'] for p in alive]
    receiver = max((i for i in range(len(alive)) if i != h), key=lambda i: alive[i]['bet'])
    baseline = current.copy()
    baseline[receiver] += pot
    u_fold = value(baseline, h, payouts)
    take = current.copy()
    take[h] += pot
    u_take = value(take, h, payouts)
    expected = fold_probability * u_take
    for seat, probability, equity in callers:
        v = seats.index(seat)
        villain = alive[v]
        total = min(me['stack'] + me['bet'], villain['stack'] + villain['bet'])
        risk = max(0, total - me['bet'])
        extra = max(0, total - villain['bet'])
        win, lose = current.copy(), current.copy()
        win[h] += pot + extra
        win[v] -= extra
        lose[h] -= risk
        lose[v] += pot + risk
        expected += probability * (equity * value(win, h, payouts) + (1 - equity) * value(lose, h, payouts))
    return expected - u_fold, max(0.0, u_take - u_fold)
