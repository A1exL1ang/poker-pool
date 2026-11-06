"""Exact independent chip model for caller-provided payouts only.

Subset DP marginalizes all opponent finishing orders. No payout schedule is
assumed; chips remain the objective when payouts are absent or linear.
"""


def equity(stacks, payouts, hero):
    """Hero expected payout, for positive live stacks and descending payouts."""
    h = float(stacks[hero])
    n = len(stacks)
    prizes = tuple(payouts[:n]) + (0.0,) * max(0, n - len(payouts))
    if h <= 0:
        return float(prizes[-1])
    others = [float(s) for i, s in enumerate(stacks) if i != hero and s > 0]
    total = h + sum(others)
    guaranteed = float(prizes[-1])
    prizes = tuple(float(p) - guaranteed for p in prizes)
    last = next((i for i in range(n - 1, -1, -1) if prizes[i] != 0), -1)
    if last < 0:
        return guaranteed
    size = 1 << len(others)
    mass = [0.0] * size
    sums = [0.0] * size
    mass[0] = 1.0
    result = guaranteed
    for mask in range(size):
        place = mask.bit_count()
        if place > last:
            continue
        if mask:
            low = mask & -mask
            sums[mask] = sums[mask ^ low] + others[low.bit_length() - 1]
        denom = total - sums[mask]
        scaled = mass[mask] / denom
        result += scaled * h * prizes[place]
        if place == last:
            continue
        remaining = (size - 1) ^ mask
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            mass[mask | bit] += scaled * others[bit.bit_length() - 1]
    return result


def call_threshold(observation, opponent):
    """Exact two-outcome ICM break-even probability for a heads-up all-in.

    Returns None unless all live hand opposition is one seat and investments
    can be settled as a single matched pot. Ties and future skill are ignored.
    Folded/dead contributions ARE included. This is a decision threshold, not
    a claim that ICM predicts tournament play. Full payouts must be supplied.
    """
    payouts = observation.get('payouts')
    if not payouts:
        return None
    players = observation['players']
    hero = observation['hero']
    if any(not p['folded'] and i not in (hero, opponent) for i, p in enumerate(players)):
        return None
    ids = [i for i, p in enumerate(players) if p['stack'] + p['invested'] > 0]
    if len(ids) > 8 or len(payouts) < len(ids):
        return None
    p, v = players[hero], players[opponent]
    call = min(observation['to_call'], p['stack'])
    if call <= 0 or (call < p['stack'] and v['stack'] > 0):
        return None
    if any(q['invested'] > p['invested'] + call for i, q in enumerate(players) if i not in (hero, opponent)):
        return None  # A side pot outside hero's eligibility needs a richer model.
    # An unmatched part of the opponent's wager is returned in both outcomes.
    refund = max(0, v['invested'] - p['invested'] - call)
    pot = observation['pot'] + call - refund
    if pot <= 0:
        return None
    ix, vx = ids.index(hero), ids.index(opponent)
    base = [players[i]['stack'] for i in ids]
    fold = base[:]
    fold[vx] += observation['pot']
    win = base[:]
    win[ix] += pot - call
    win[vx] += refund
    lose = base[:]
    lose[ix] -= call
    lose[vx] += observation['pot'] + call
    # Remaining prizes start at first place; already busted seats took bottom.
    prizes = payouts[:len(ids)]
    def utility(state):
        if state[ix] <= 0:
            return float(prizes[len(ids) - 1])
        live = [j for j, value in enumerate(state) if value > 0]
        return equity([state[j] for j in live], prizes[:len(live)], live.index(ix))
    u_fold, u_win, u_lose = utility(fold), utility(win), utility(lose)
    return max(0.0, min(1.0, (u_fold - u_lose) / max(1e-12, u_win - u_lose)))
