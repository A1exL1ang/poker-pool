"""Small exact ICM calculator. Payouts are supplied by the caller, never assumed.

The subset dynamic program computes only the hero's equity, O(n 2**(n-1)).
All payout arrays are ordered first place to last place in the whole table.
"""
_SUBSETS = []
for _n in range(8):
    _entries = []
    for _mask in range(1 << _n):
        _links = tuple((_mask ^ (1 << i), i) for i in range(_n) if _mask & (1 << i))
        _entries.append((_mask.bit_count(), _links))
    _SUBSETS.append(tuple(_entries))


def equity(stacks, hero, payouts):
    """Return hero's ICM points; zero stacks share the remaining bottom places.

    This function describes a terminal distribution, rather than resolving the
    ordering of simultaneous eliminations. The local engine handles that order.
    """
    n = len(stacks)
    prizes = tuple(float(x) for x in payouts[:n]) + (0.0,) * max(0, n - len(payouts))
    live = [i for i, x in enumerate(stacks) if x > 0]
    if stacks[hero] <= 0:
        return sum(prizes[len(live):]) / max(1, n - len(live))
    h = float(stacks[hero])
    others = tuple(float(stacks[i]) for i in live if i != hero)
    m = len(others)
    base = prizes[m]
    # Equal trailing payouts need no further recursive enumeration.
    last = m
    while last and prizes[last - 1] == base:
        last -= 1
    if not last:
        return base
    total = h + sum(others)

    # Scaled reach[mask] is the probability that exactly this set of opponents
    # finishes above hero, divided by the remaining stack total. Subsets merge
    # all their permutations, with no recursive calls in the timed hot path.
    scaled = [0.] * (1 << m)
    removed = [0.] * (1 << m)
    scaled[0] = 1. / total
    value = base + h / total * (prizes[0] - base)
    for mask, (depth, links) in enumerate(_SUBSETS[m][1:], 1):
        if depth >= last:
            continue
        previous, i = links[0]
        removed[mask] = removed[previous] + others[i]
        reach = 0.
        for previous, i in links:
            reach += others[i] * scaled[previous]
        scaled[mask] = reach / (total - removed[mask])
        value += h * scaled[mask] * (prizes[depth] - base)
    return value


def call_threshold(observation, villain, cost, eligible_pot, villain_add=0):
    """HU ICM break-even showdown equity for a call (ties approximated by equity).

    Exact when the current hand has two live contestants and no side pots;
    multiway users should retain a conservative chip-EV threshold instead.
    Unknown payouts imply linear chip value. Sunk investments are not charged
    again. `eligible_pot` includes all currently contestable chips before call.
    """
    cost = max(0.0, float(cost))
    pot = max(0.0, float(eligible_pot))
    chip_threshold = cost / max(1.0, pot + cost)
    payouts = observation.get("payouts")
    if not payouts or cost <= 0:
        return chip_threshold
    players = observation["players"]
    hero = observation["hero"]
    # Already eliminated seats have fixed points and must not be pooled with a
    # player eliminated by this call, especially when lower places pay points.
    active = [i for i, p in enumerate(players)
              if p.get("starting_stack", p["stack"] + p["invested"]) > 0]
    index = {seat: j for j, seat in enumerate(active)}
    hero, villain = index[hero], index[villain]
    stacks = [float(players[i]["stack"]) for i in active]
    payouts = payouts[:len(active)]
    stacks[villain] -= villain_add
    # Assign non-contestable side money to the aggressor in all three cases.
    stacks[villain] += max(0.0, observation["pot"] + villain_add - pot)
    fold = stacks[:]
    fold[villain] += pot
    lose = stacks[:]
    lose[hero] -= cost
    lose[villain] += pot + cost
    win = stacks[:]
    win[hero] += pot
    vf = equity(fold, hero, payouts)
    vl = equity(lose, hero, payouts)
    vw = equity(win, hero, payouts)
    if vw <= vl:
        return 1.0
    return max(chip_threshold, min(0.995, (vf - vl) / (vw - vl)))
