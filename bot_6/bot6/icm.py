"""Exact independent chip model; no assumed contest payout vector."""


def equity(stacks, payouts, hero):
    if len(stacks) != len(payouts) or not stacks:
        raise ValueError("Stacks and payouts must have equal positive length")
    if any(s < 0 for s in stacks) or any(a < b for a, b in zip(payouts, payouts[1:])):
        raise ValueError("Invalid stacks or descending payouts")
    h = stacks[hero]
    if h <= 0:
        # Equal zero stacks split the tied last places; elimination order is
        # otherwise unavailable and must be resolved by the caller.
        nz = sum(s > 0 for s in stacks)
        return sum(payouts[nz:]) / (len(stacks) - nz)
    others = [s for i, s in enumerate(stacks) if i != hero and s > 0]
    n = len(others)
    total = h + sum(others)
    baseline = payouts[n]
    last_rank = next((i for i in range(n - 1, -1, -1) if payouts[i] > baseline), -1)
    if last_rank < 0:
        return baseline
    if last_rank == 0:
        return baseline + (payouts[0] - baseline) * h / total
    sums = [0.] * (1 << n)
    probabilities = [0.] * (1 << n)
    probabilities[0] = 1.
    result = baseline
    for mask in range(1 << n):
        if mask:
            bit = mask & -mask
            sums[mask] = sums[mask ^ bit] + others[bit.bit_length() - 1]
        rank = mask.bit_count()
        if rank > last_rank:
            continue
        remaining = total - sums[mask]
        p = probabilities[mask]
        result += p * h / remaining * (payouts[rank] - baseline)
        if rank == last_rank:
            continue
        for j, stack in enumerate(others):
            bit = 1 << j
            if not mask & bit:
                probabilities[mask | bit] += p * stack / remaining
    return result


def risk_premium(state):
    """Conservative payout-aware heuristic; this is NOT exact ICM strategy.

    Exact ICM is provided above for offline scoring. Unknown payouts imply
    zero premium. Winner-take-all is linear in chips and also gets zero.
    """
    if state.risk_premium:
        return state.risk_premium
    p = state.payouts
    if not p or len(p) <= 2 or p[0] == p[-1]:
        return 0.
    spread = p[0] - p[-1]
    tail = sum(x - p[-1] for x in p[1:])
    if tail <= 0:
        return 0.
    me = state.me
    opponents = state.opponents
    covered = any(v.stack + v.bet >= me.stack + me.bet for v in opponents)
    shorter = sum(0 < v.stack + v.bet < (me.stack + me.bet) * .55 for v in state.players if v.seat != state.hero)
    paid = sum(x > p[-1] for x in p)
    bubble = 1. if len(p) == paid + 1 else .6
    base = min(.085, .025 * tail / spread) * bubble
    return min(.12, base * (1.2 + .25 * shorter) if covered else base * .4)


def allin_call_threshold(state, contestable):
    """Exact ICM win-probability threshold for a heads-up all-in decision.

    Other seated players may have folded. Requires complete contributions.
    A fractional showdown-equity comparison approximates ties; this is not
    a future-game solver. Return None outside the supported case.
    """
    opponents = state.opponents
    if not state.payouts or len(opponents) != 1 or not opponents[0].all_in or state.to_call <= 0:
        return None
    if sum(p.contributed for p in state.players) != state.pot:
        return None
    hero = next(i for i, p in enumerate(state.players) if p.seat == state.hero)
    villain = next(i for i, p in enumerate(state.players) if p.seat == opponents[0].seat)
    stacks = [p.stack for p in state.players]
    fold, win, lose = stacks[:], stacks[:], stacks[:]
    fold[villain] += state.pot
    eligible = contestable - state.to_call
    win[hero] += eligible
    win[villain] += state.pot - eligible
    lose[hero] -= state.to_call
    lose[villain] += state.pot + state.to_call
    u_fold = equity(fold, state.payouts, hero)
    u_win = equity(win, state.payouts, hero)
    u_lose = equity(lose, state.payouts, hero)
    if u_win <= u_lose:
        return 1.
    return max(0., min(1., (u_fold - u_lose) / (u_win - u_lose)))
