"""Independent, local no-limit Hold'em simulator; NOT the official starter kit.

Cards use rank-major 0..51. Dealing omits burns (distribution-equivalent when
burns are unobserved). Default blind progression is a synthetic test assumption.
Payouts must be supplied explicitly for point/ICM scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from random import Random
from time import perf_counter_ns
from typing import Callable, Sequence


def rank_five(cards: Sequence[int]) -> tuple:
    """Slow independent 5-card reference evaluator; larger tuple is stronger."""
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    counts = {r: ranks.count(r) for r in set(ranks)}
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    unique = sorted(counts, reverse=True)
    straight = 5 if unique == [14, 5, 4, 3, 2] else (unique[0] if len(unique) == 5 and unique[0] - unique[-1] == 4 else 0)
    flush = len({c % 4 for c in cards}) == 1
    if straight and flush:
        return (8, straight)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if [g[0] for g in groups] == [3, 2]:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *ranks)
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        return (3, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
    if groups[0][0] == 2 and groups[1][0] == 2:
        return (2, max(groups[0][1], groups[1][1]), min(groups[0][1], groups[1][1]), groups[2][1])
    if groups[0][0] == 2:
        return (1, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
    return (0, *ranks)


def evaluate_reference(cards: Sequence[int]) -> tuple:
    if len(cards) not in (5, 6, 7) or len(set(cards)) != len(cards):
        raise ValueError('need five to seven distinct cards')
    return max(rank_five(c) for c in combinations(cards, 5))


def evaluate(cards: Sequence[int]) -> tuple:
    """Fast independently implemented direct evaluator for simulator showdowns."""
    counts = [0] * 15
    suits = [[], [], [], []]
    for card in cards:
        rank, suit = card // 4 + 2, card % 4
        counts[rank] += 1
        suits[suit].append(rank)
    ranks = [r for r in range(14, 1, -1) if counts[r]]
    def straight(rs):
        mask = sum(1 << r for r in rs)
        if mask & (1 << 14):
            mask |= 2
        for top in range(14, 4, -1):
            if (mask >> (top - 4)) & 31 == 31:
                return top
        return 0
    flush = next((sorted(s, reverse=True) for s in suits if len(s) >= 5), None)
    if flush:
        sf = straight(flush)
        if sf:
            return (8, sf)
    quads = [r for r in ranks if counts[r] == 4]
    if quads:
        return (7, quads[0], next(r for r in ranks if r != quads[0]))
    trips = [r for r in ranks if counts[r] >= 3]
    if trips:
        pairs = [r for r in ranks if r != trips[0] and counts[r] >= 2]
        if pairs:
            return (6, trips[0], pairs[0])
    if flush:
        return (5, *flush[:5])
    st = straight(ranks)
    if st:
        return (4, st)
    if trips:
        return (3, trips[0], *[r for r in ranks if r != trips[0]][:2])
    pairs = [r for r in ranks if counts[r] >= 2]
    if len(pairs) >= 2:
        return (2, pairs[0], pairs[1], next(r for r in ranks if r not in pairs[:2]))
    if pairs:
        return (1, pairs[0], *[r for r in ranks if r != pairs[0]][:3])
    return (0, *ranks[:5])


def icm_values(stacks: Sequence[int | float], payouts: Sequence[float]) -> list[float]:
    """Exact subset dynamic-programming ICM for living players, O(n^2 2^n).

    Payouts are explicit local inputs. Zero stacks receive zero; call separately
    for eliminated places. Equal prizes and winner-take-all are valid.
    """
    n = len(stacks)
    if len(payouts) < sum(s > 0 for s in stacks):
        raise ValueError('not enough payouts for living players')
    alive = [i for i, s in enumerate(stacks) if s > 0]
    m = len(alive)
    result = [0.0] * n
    if not m:
        return result
    values = [float(stacks[i]) for i in alive]
    total = sum(values)
    masses = [0.0] * (1 << m)
    masses[0] = 1.0
    selected = [0.0] * (1 << m)
    for mask in range(1, 1 << m):
        low = mask & -mask
        selected[mask] = selected[mask ^ low] + values[low.bit_length() - 1]
    for mask in range((1 << m) - 1):
        denominator = total - selected[mask]
        place = mask.bit_count()
        for j in range(m):
            bit = 1 << j
            if not mask & bit:
                probability = masses[mask] * values[j] / denominator
                result[alive[j]] += probability * payouts[place]
                masses[mask | bit] += probability
    return result


@dataclass
class HandResult:
    stacks: list[int]
    history: list[dict]
    board: list[int]
    holes: list[list[int]]
    contributions: list[int]
    awards: list[int]
    refunds: list[int]
    violations: list[dict] = field(default_factory=list)
    timings_ns: list[int] = field(default_factory=list)
    decisions_by_seat: list[int] = field(default_factory=list)
    overruns_by_seat: list[int] = field(default_factory=list)


@dataclass
class TableResult:
    stacks: list[int]
    points: list[float] | None
    places: list[float | None]
    hands: int
    violations: list[dict]
    timings_ns: list[int]
    decisions_by_seat: list[int]
    overruns_by_seat: list[int]
    elimination_groups: list[list[int]]


def _next(seat, seats):
    n = len(seats)
    for offset in range(1, n + 1):
        j = (seat + offset) % n
        if seats[j]:
            return j
    raise ValueError('no live seats')


def play_hand(agents, stacks, button=0, small_blind=10, big_blind=20,
              seed=0, hand_id=0, deck=None, observe: Callable | None = None,
              payouts=None, hands_remaining=None, enforce_time_limit=False,
              decision_limit_ns=1_000_000) -> HandResult:
    """Play a hand with defensive legal-action validation.

    `observe(seat, observation, action, duration_ns)` receives each completed
    decision; observation contains only that seat's private cards. Returned
    HandResult is omniscient and must never be passed to live agents.
    Timeouts/errors cause check/fold and are recorded; timing is measured even
    when enforcement is disabled for strategic benchmarking.
    """
    n = len(stacks)
    if len(agents) != n or n < 2:
        raise ValueError('need one agent per seat, and at least two seats')
    if any(type(s) is not int or s < 0 for s in stacks):
        raise ValueError('stacks must be nonnegative integer chips')
    if not (type(big_blind) is int and type(small_blind) is int and 0 <= small_blind <= big_blind and big_blind > 0):
        raise ValueError('invalid blinds')
    initial_total = sum(stacks)
    players = [dict(stack=s, street_bet=0, invested=0, folded=(s == 0), all_in=False) for s in stacks]
    live = [s > 0 for s in stacks]
    if sum(live) < 2:
        raise ValueError('need at least two funded players')
    button %= n
    if not live[button]:
        button = _next(button, live)
    rng = Random(seed)
    if deck is None:
        deck = list(range(52))
        rng.shuffle(deck)
    else:
        deck = list(deck)
        if len(deck) != 52 or set(deck) != set(range(52)):
            raise ValueError('deck must be a permutation of 0..51')
    cursor = 0
    holes = [[] for _ in players]
    dealing_order = []
    seat = button
    for _ in range(sum(live)):
        seat = _next(seat, live)
        dealing_order.append(seat)
    for _ in range(2):
        for seat in dealing_order:
            holes[seat].append(deck[cursor])
            cursor += 1
    headsup = sum(live) == 2
    sb = button if headsup else _next(button, live)
    bb = _next(sb, live)
    def pay(seat, amount):
        amount = min(amount, players[seat]['stack'])
        p = players[seat]
        p['stack'] -= amount
        p['street_bet'] += amount
        p['invested'] += amount
        p['all_in'] = p['stack'] == 0
        return amount
    pay(sb, small_blind)
    pay(bb, big_blind)
    history, board, violations, timings = [], [], [], []
    decisions, overruns = [0] * n, [0] * n
    for street in range(4):
        if sum(not p['folded'] for p in players) == 1:
            break
        if street:
            draw = 3 if street == 1 else 1
            board.extend(deck[cursor:cursor + draw])
            cursor += draw
            for p in players:
                p['street_bet'] = 0
        actionable = {i for i, p in enumerate(players) if not p['folded'] and p['stack'] > 0}
        current_bet = (big_blind if street == 0 and len(actionable) >= 2
                       else max(p['street_bet'] for p in players))
        last_full_raise = big_blind
        acted_at = [None] * n
        checked = [False] * n
        pending = set(actionable)
        actor = (button if headsup else _next(bb, live)) if street == 0 else _next(button, live)
        safety = 0
        while pending:
            if sum(not p['folded'] for p in players) <= 1:
                break
            actionable = {i for i, p in enumerate(players) if not p['folded'] and p['stack'] > 0}
            pending.intersection_update(actionable)
            if len(actionable) == 1:
                only = next(iter(actionable))
                actual_to_match = max(p['street_bet'] for p in players if not p['folded'])
                current_bet = actual_to_match
                if players[only]['street_bet'] >= actual_to_match:
                    break
            if not pending:
                break
            if actor not in pending:
                actor = (actor + 1) % n
                continue
            p = players[actor]
            to_call = max(0, current_bet - p['street_bet'])
            max_to = p['street_bet'] + p['stack']
            opponents_can_respond = any(i != actor and not q['folded'] and q['stack'] > 0 and q['street_bet'] + q['stack'] > current_bet for i, q in enumerate(players))
            reopened = acted_at[actor] is None or current_bet - acted_at[actor] >= last_full_raise
            can_raise = opponents_can_respond and reopened and max_to > current_bet
            obs = dict(hero=actor, button=button, hand_id=hand_id, street=street,
                       hole=list(holes[actor]), board=list(board),
                       players=[dict(q) for q in players], pot=sum(q['invested'] for q in players),
                       to_call=to_call, big_blind=big_blind, small_blind=small_blind,
                       min_raise_to=current_bet + last_full_raise,
                       max_raise_to=max_to, can_raise=can_raise,
                       history=[dict(a) for a in history])
            if payouts is not None:
                obs['payouts'] = list(payouts)
            if hands_remaining is not None:
                obs['hands_remaining'] = hands_remaining
            fallback = {'action': 'check' if to_call == 0 else 'fold'}
            started = perf_counter_ns()
            error = None
            try:
                action = agents[actor].act(obs)
            except Exception as exc:
                action = fallback
                error = f'exception: {type(exc).__name__}: {exc}'
            elapsed = perf_counter_ns() - started
            timings.append(elapsed)
            decisions[actor] += 1
            if elapsed > decision_limit_ns:
                overruns[actor] += 1
                if enforce_time_limit:
                    action = fallback
                    error = 'decision timeout'
            if not isinstance(action, dict) or action.get('action') not in ('fold', 'check', 'call', 'raise'):
                error = error or 'invalid action object'
            else:
                kind = action['action']
                if kind == 'check' and to_call > 0:
                    error = error or 'check facing wager'
                elif kind == 'call' and to_call == 0:
                    error = error or 'call with nothing to call'
                elif kind == 'raise':
                    amount = action.get('amount')
                    if type(amount) is not int or not can_raise or amount <= current_bet or amount > max_to or (amount < current_bet + last_full_raise and amount != max_to):
                        error = error or 'illegal raise total or reopening'
            if error:
                violations.append(dict(seat=actor, street=street, hand_id=hand_id, error=error, proposed=repr(action)))
                action = fallback
            if observe is not None:
                observe(actor, obs, dict(action), elapsed)
            kind = action['action']
            before = sum(q['invested'] for q in players)
            pending.discard(actor)
            paid = 0
            if kind == 'fold':
                p['folded'] = True
            elif kind == 'call':
                paid = pay(actor, to_call)
            elif kind == 'raise':
                target = action['amount']
                increase = target - current_bet
                paid = pay(actor, target - p['street_bet'])
                if increase >= last_full_raise:
                    last_full_raise = increase
                current_bet = target
                pending.update(i for i, q in enumerate(players) if i != actor and not q['folded'] and q['stack'] > 0 and q['street_bet'] < current_bet)
            acted_at[actor] = current_bet
            checked[actor] = kind == 'check'
            history.append(dict(seat=actor, street=street, action=kind, paid=paid, to=p['street_bet'], pot_before=before))
            safety += 1
            if safety > 10000:
                raise RuntimeError('betting failed to terminate')
            actor = (actor + 1) % n
    # Refund the unique uncalled highest contribution, including a folded
    # player's contribution when finding the second-highest amount.
    refunds = [0] * n
    contributions = [p['invested'] for p in players]
    largest = max(contributions)
    leaders = [i for i, value in enumerate(contributions) if value == largest]
    if len(leaders) == 1:
        leader = leaders[0]
        second = max(value for i, value in enumerate(contributions) if i != leader)
        refund = largest - second
        if not players[leader]['folded']:
            refunds[leader] = refund
            players[leader]['stack'] += refund
            players[leader]['invested'] -= refund
            contributions[leader] -= refund
    survivors = [i for i, p in enumerate(players) if not p['folded']]
    awards = [0] * n
    if len(survivors) == 1:
        awards[survivors[0]] = sum(contributions)
    else:
        if len(board) != 5:
            raise AssertionError('showdown without five board cards')
        ranks = {i: evaluate(holes[i] + board) for i in survivors}
        previous = 0
        for level in sorted(set(contributions) - {0}):
            contributors = [i for i, value in enumerate(contributions) if value >= level]
            pot = (level - previous) * len(contributors)
            eligible = [i for i in contributors if not players[i]['folded']]
            if not eligible:
                raise AssertionError('side pot without eligible player')
            best = max(ranks[i] for i in eligible)
            winners = [i for i in eligible if ranks[i] == best]
            share, odd = divmod(pot, len(winners))
            for i in winners:
                awards[i] += share
            odd_order = sorted(winners, key=lambda i: (i - button - 1) % n)
            for i in odd_order[:odd]:
                awards[i] += 1
            previous = level
    final = [p['stack'] + awards[i] for i, p in enumerate(players)]
    if any(s < 0 for s in final) or sum(final) != initial_total:
        raise AssertionError('chip conservation violated')
    return HandResult(final, history, board, holes, contributions, awards, refunds,
                      violations, timings, decisions, overruns)


# Explicitly SYNTHETIC; only the first 10/20 level and 25-hand interval are
# specified in supplied rules. Changing later levels is an experiment parameter.
SYNTHETIC_BLINDS = [(10, 20), (15, 30), (25, 50), (40, 80),
                    (60, 120), (100, 200), (150, 300), (250, 500)]


def play_table(agents, starting_stack=2000, max_hands=200, seed=0,
               blind_schedule=None, payouts=None, observe=None,
               enforce_time_limit=False, button=None) -> TableResult:
    """Local sit-and-go with forward-moving button and corrected HU transition.

    Forward-moving and dead-button systems are both used by poker platforms;
    the supplied competition text does not specify which. This local simulator
    uses the former for 3+ players and prevents consecutive BB at HU transition.
    """
    n = len(agents)
    stacks = [starting_stack] * n if type(starting_stack) is int else list(starting_stack)
    if len(stacks) != n:
        raise ValueError('stack count mismatch')
    if payouts is not None and len(payouts) != n:
        raise ValueError('need one explicit payout per place')
    if payouts is not None and any(payouts[i] < payouts[i + 1] for i in range(n - 1)):
        raise ValueError('payouts must be nonincreasing')
    blinds = list(blind_schedule or SYNTHETIC_BLINDS)
    rng = Random(seed)
    if button is None:
        button = rng.randrange(n)
    for i, agent in enumerate(agents):
        if hasattr(agent, 'reset'):
            agent.reset(seed=seed * 1009 + i * 9176 + 16)
    places = [None] * n
    points = [0.0] * n if payouts is not None else None
    violations, timings, groups = [], [], []
    decisions, overruns = [0] * n, [0] * n
    hands = 0
    for hand_id in range(max_hands):
        living = [s > 0 for s in stacks]
        if sum(living) <= 1:
            break
        if not living[button]:
            button = _next(button, living)
        sb, bb = blinds[min(hand_id // 25, len(blinds) - 1)]
        before = list(stacks)
        previous_sb = button if sum(living) == 2 else _next(button, living)
        previous_bb = _next(previous_sb, living)
        result = play_hand(agents, stacks, button=button, small_blind=sb, big_blind=bb,
                           seed=rng.getrandbits(64), hand_id=hand_id, observe=observe,
                           payouts=payouts, hands_remaining=max_hands - hand_id,
                           enforce_time_limit=enforce_time_limit)
        stacks = result.stacks
        violations.extend(result.violations)
        timings.extend(result.timings_ns)
        decisions = [a + b for a, b in zip(decisions, result.decisions_by_seat)]
        overruns = [a + b for a, b in zip(overruns, result.overruns_by_seat)]
        eliminated = [i for i in range(n) if before[i] > 0 and stacks[i] == 0]
        remaining = sum(s > 0 for s in stacks)
        # Standard local tie convention: larger stack at start of simultaneous
        # bust-out receives better place; equal starting stacks split the points.
        # The supplied official text does not specify this tie convention.
        place = remaining + 1
        for size in sorted({before[i] for i in eliminated}, reverse=True):
            group = [i for i in eliminated if before[i] == size]
            mean_place = sum(range(place, place + len(group))) / len(group)
            for i in group:
                places[i] = mean_place
                if points is not None:
                    points[i] = sum(payouts[place - 1:place - 1 + len(group)]) / len(group)
            groups.append(group)
            place += len(group)
        hands += 1
        survivors = [s > 0 for s in stacks]
        if sum(survivors) == 2:
            # Heads-up transition must not make the same player post BB twice.
            next_bb = _next(previous_bb, survivors)
            button = _next(next_bb, survivors)
        elif sum(survivors) > 2:
            # Local online-style forward-moving button convention.
            button = _next(button, survivors)
    living = [i for i, s in enumerate(stacks) if s > 0]
    if len(living) == 1:
        places[living[0]] = 1.0
    if points is not None and living:
        icm = icm_values(stacks, payouts[:len(living)])
        for i in living:
            points[i] = icm[i]
        if abs(sum(points) - sum(payouts)) > 1e-8:
            raise AssertionError('points not conserved')
    return TableResult(stacks, points, places, hands, violations, timings,
                       decisions, overruns, groups)

