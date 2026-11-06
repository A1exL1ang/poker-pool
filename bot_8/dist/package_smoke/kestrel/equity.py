"""Offline preflop lookup and bounded, range-weighted postflop quadrature.

Runtime samples are deterministic for a decision. There are no network calls,
native extensions, unbounded rejection loops, or external runtime dependencies.
The opponent range model is deliberately a heuristic, not a solved equilibrium.
"""

import json
from pathlib import Path
from math import sqrt
from time import perf_counter_ns

from .cards import BITS, EMPTY, extend, score, hand_class

with Path(__file__).with_name("preflop.json").open(encoding="utf-8") as _f:
    _data = json.load(_f)
FRACTIONS = tuple(_data["fractions"])
BASE = tuple(_data["base"])
EQUITY = tuple(tuple(row) for row in _data["equity"])
ALL_COMBOS = tuple(sorted(((a, b) for a in range(52) for b in range(a + 1, 52)),
                          key=lambda h: BASE[hand_class(*h)], reverse=True))
_pools = []
for _fraction in FRACTIONS:
    _last = min(1325, max(5, int(1326 * _fraction + .5) - 1))
    _cutoff = BASE[hand_class(*ALL_COMBOS[_last])]
    _pools.append(tuple(h for h in ALL_COMBOS if BASE[hand_class(*h)] >= _cutoff))
POOLS = tuple(_pools)
del _pools
_counts = [0] * 169
for _a, _b in ALL_COMBOS:
    _counts[hand_class(_a, _b)] += 1
PERCENTILE = [0.0] * 169
_cum = 0
for _c in sorted(range(169), key=lambda c: BASE[c], reverse=True):
    PERCENTILE[_c] = (_cum + .5 * _counts[_c]) / 1326
    _cum += _counts[_c]
PERCENTILE = tuple(PERCENTILE)
del _data, _counts

_WINDOWS = (0x100F, 31, 62, 124, 248, 496, 992, 1984, 3968, 7936)
STRAIGHT_DRAW = tuple(any((m & w).bit_count() == 4 for w in _WINDOWS) for m in range(8192))


def board_context(board):
    counts = [0] * 13
    suits = [0] * 4
    mask = 0
    for c in board:
        counts[c >> 2] += 1
        suits[c & 3] += 1
        mask |= BITS[c]
    ranks = [r for r in range(13) if counts[r]]
    flush_suit = max(range(4), key=suits.__getitem__)
    return (tuple(counts), tuple(suits), mask, ranks[-1], ranks[-2] if len(ranks) >= 2 else ranks[-1],
            flush_suit, suits[flush_suit] >= 4, len(board))


def fast_quality(hole, context, value):
    counts, suits, mask, high, second, flush_suit, four_flush, length = context
    a, b = hole[0] >> 2, hole[1] >> 2
    category = value >> 20
    if category >= 4:
        if category >= 7:
            return .995
        if category == 4 and four_flush:
            return .70
        if category == 5:
            if not four_flush:
                return .975
            own = max(a if hole[0] & 3 == flush_suit else -1, b if hole[1] & 3 == flush_suit else -1)
            return .74 + .018 * max(0, own)
        return .965 if category == 4 else .989
    if category == 3:
        count = counts[((value >> 16) & 15) - 2]
        return .965 if count == 1 else .90 if count == 2 else .44 + .023 * max(a, b)
    if category == 2:
        pa, pb = ((value >> 16) & 15) - 2, ((value >> 12) & 15) - 2
        own_pairs = (counts[pa] == 1) + (counts[pb] == 1)
        if own_pairs == 2:
            return .91
        if own_pairs == 1:
            return .83 if max(a, b) == high else .70
        if a == b:
            return .83 if a > high else .64
        return .34 + .023 * max(a, b)
    if category == 1:
        pair = ((value >> 16) & 15) - 2
        if a == b:
            return .83 if a > high else .57 + .012 * a
        if pair in (a, b) and counts[pair] == 1:
            kicker = b if a == pair else a
            return .72 + .009 * kicker if pair == high else .54 + .007 * kicker if pair >= second else .45 + .006 * kicker
        return .25 + .025 * max(a, b)
    return .12 + .032 * max(a, b)


def fast_draw(hole, context):
    counts, suits, mask, high, second, flush_suit, four_flush, length = context
    if length == 5:
        return 0.0
    sa, sb = hole[0] & 3, hole[1] & 3
    if suits[sa] + 1 + (sa == sb) == 4 or suits[sb] + 1 + (sa == sb) == 4:
        return .26
    return .14 if STRAIGHT_DRAW[mask | BITS[hole[0]] | BITS[hole[1]]] else 0.0


def preflop_equity(hand, width):
    row = EQUITY[hand]
    if width <= FRACTIONS[0]:
        return row[0]
    for j in range(1, len(FRACTIONS)):
        if width <= FRACTIONS[j]:
            t = (width - FRACTIONS[j - 1]) / (FRACTIONS[j] - FRACTIONS[j - 1])
            return row[j - 1] * (1 - t) + row[j] * t
    return row[-1]


def pool_index(width):
    return min(range(8), key=lambda j: abs(FRACTIONS[j] - width))


def made_quality(hole, board, value):
    """Approximate present made-hand percentile, used ONLY for action likelihoods."""
    category = value >> 20
    ranks = tuple(c >> 2 for c in board)
    a, b = hole[0] >> 2, hole[1] >> 2
    high = max(ranks)
    # On a five-card board the board's own hand is sometimes most of the value.
    if category >= 4:
        if category >= 7:
            return .995
        suits = [0, 0, 0, 0]
        for c in board:
            suits[c & 3] += 1
        four_flush = max(suits) >= 4
        if category == 4 and four_flush:
            return .70
        if category == 5:
            suit = max(range(4), key=suits.__getitem__)
            own = max((c >> 2 for c in hole if (c & 3) == suit), default=-1)
            return .74 + .018 * max(0, own) if four_flush else .975
        return .965 if category == 4 else .989
    if category == 3:
        t = ((value >> 16) & 15) - 2
        count = ranks.count(t)
        return .965 if count == 1 else .90 if count == 2 else .44 + .023 * max(a, b)
    if category == 2:
        pa, pb = ((value >> 16) & 15) - 2, ((value >> 12) & 15) - 2
        own_pairs = int(ranks.count(pa) == 1) + int(ranks.count(pb) == 1)
        if own_pairs == 2:
            return .91
        if own_pairs == 1:
            return .83 if max(a, b) == high else .70
        if a == b:
            return .83 if a > high else .64
        return .34 + .023 * max(a, b)
    if category == 1:
        pair = ((value >> 16) & 15) - 2
        if a == b:
            return .83 if a > high else .57 + .012 * a
        if pair in (a, b) and ranks.count(pair) == 1:
            kicker = b if a == pair else a
            return (.72 + .009 * kicker if pair == high else
                    .54 + .007 * kicker if pair >= sorted(set(ranks))[-2] else .45 + .006 * kicker)
        return .25 + .025 * max(a, b)
    return .12 + .032 * max(a, b)


def _draw_likelihood(hole, board):
    if len(board) == 5:
        return 0.0
    suits = [0, 0, 0, 0]
    mask = 0
    for c in board:
        suits[c & 3] += 1
        mask |= BITS[c]
    for c in hole:
        suits[c & 3] += 1
        mask |= BITS[c]
    if any(suits[c & 3] == 4 for c in hole):
        return .26
    # An open-ended draw or gutshot makes semibluff actions plausible.
    for shift in range(9):
        if (mask & (31 << shift)).bit_count() == 4:
            return .14
    return .0


def estimate(hole, board, profiles, seed, sample_scale=1.0, deadline_ns=None, pots=None):
    """Return (showdown equity, effective samples, standard error).

    Profiles are (range-pool index, aggression, passivity, bluff-prior).
    Each sample jointly deals ALL active opponents without duplicate cards.
    Aggression/passivity reweight present-street made hands and draws. Past
    streets only contribute softened evidence; this is not Bayesian perfection.
    """
    n = len(profiles)
    if not n:
        return 1.0, 100.0, 0.0
    sample_count = max(8, int((64 if n == 1 else 40 if n == 2 else 28 if n == 3 else 20) * sample_scale))
    if len(board) == 5 and n <= 2:
        sample_count += int(16 * sample_scale)
    base_state = extend(EMPTY, board)
    context = board_context(board)
    current_hero = score(extend(base_state, hole))
    blocked = sum(1 << c for c in hole + board)
    board_score = score(base_state) if len(board) == 5 else -1
    prepared = tuple((POOLS[p[0]], p[1], p[2], p[3]) for p in profiles)
    total = wins = weight2 = 0.0
    rng = (seed ^ 0xA3817F29) & 0xFFFFFFFF
    for sample in range(sample_count):
        # The timer is a guard against slow hardware/large histories, not a
        # claim of a hard real-time guarantee under operating-system preemption.
        if sample and deadline_ns is not None and perf_counter_ns() >= deadline_ns:
            break
        used = blocked
        hands = []
        weight = 1.0
        for pool, aggression, passive, bluff_prior in prepared:
            accepted = False
            for attempt in range(10):
                rng = (1664525 * rng + 1013904223) & 0xFFFFFFFF
                a, b = pool[(rng >> 8) % len(pool)]
                bits = (1 << a) | (1 << b)
                if not used & bits:
                    accepted = True
                    break
            if not accepted:
                # Bounded deterministic fallback when tight ranges collide.
                start = (rng >> 12) % 52
                a = next(c for j in range(52) if not used & (1 << (c := (start + j) % 52)))
                b = next(c for j in range(1, 53) if not (used | (1 << a)) & (1 << (c := (a + j) % 52)))
                bits = (1 << a) | (1 << b)
                weight *= .5
            used |= bits
            h = (a, b)
            hands.append(h)
            if aggression or passive:
                value = score(extend(base_state, h))
                strength = fast_quality(h, context, value)
                if len(board) == 5 and value == board_score:
                    strength = .27
                if aggression:
                    draw = fast_draw(h, context)
                    likelihood = bluff_prior + (1.0 - bluff_prior) * strength ** (2.0 + aggression)
                    weight *= min(1.0, likelihood + draw)
                if passive:
                    weight *= (.28 + .72 * strength) ** passive
        if len(board) < 5:
            runout = []
            for _ in range(5 - len(board)):
                for attempt in range(10):
                    rng = (1664525 * rng + 1013904223) & 0xFFFFFFFF
                    c = (rng >> 8) % 52
                    if not used & (1 << c):
                        break
                else:
                    c = next(c for c in range(52) if not used & (1 << c))
                used |= 1 << c
                runout.append(c)
            state = extend(base_state, runout)
            hero = score(extend(state, hole))
        else:
            state, hero = base_state, current_hero
        if pots is None:
            share = 1.0
            ties = 1
            for h in hands:
                villain = score(extend(state, h))
                if villain > hero:
                    share = 0.0
                    break
                if villain == hero:
                    ties += 1
            if share:
                share /= ties
        else:
            values = [score(extend(state, h)) for h in hands]
            share = 0.0
            for fraction, contestants in pots:
                ties = 1
                for idx in contestants:
                    if values[idx] > hero:
                        break
                    ties += values[idx] == hero
                else:
                    share += fraction / ties
        total += weight
        wins += weight * share
        weight2 += weight * weight
    equity = wins / total if total else .5 ** n
    effective = total * total / weight2 if weight2 else 1.0
    se = sqrt(max(.005, equity * (1 - equity)) / max(effective, 1))
    return equity, effective, se
