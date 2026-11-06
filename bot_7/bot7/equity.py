"""Bounded range-conditioned sampling; no online service or native extension."""
import json
from time import perf_counter_ns
from pathlib import Path
from .cards import hand_class
from .evaluator import ENC, evaluate_sum

with open(Path(__file__).resolve().parent.parent / "assets" / "preflop.json") as f:
    _asset = json.load(f)
PREFLOP = _asset["hands"]
FRACTIONS = _asset["metadata"]["fractions"]
COMBOS = sorted(((a, b) for a in range(52) for b in range(a + 1, 52)),
                key=lambda ab: PREFLOP[hand_class(*ab)]["percentile"])
COMBO_MASK = tuple((1 << a) | (1 << b) for a, b in COMBOS)
COMBO_SUM = tuple(ENC[a] + ENC[b] for a, b in COMBOS)
COMBO_RANKS = tuple((1 << (a // 4)) | (1 << (b // 4)) for a, b in COMBOS)
del _asset


def _straight_outs(mask):
    missing = 0
    for window in (0x100F,) + tuple(31 << i for i in range(9)):
        if (mask & window).bit_count() == 4:
            missing |= window & ~mask
    return missing.bit_count()


STRAIGHT_OUTS = tuple(_straight_outs(mask) for mask in range(8192))


def preflop_equity(a, b, fraction=1.0):
    values = PREFLOP[hand_class(a, b)]["equity"]
    fraction = max(.025, min(1.0, fraction))
    for i in range(len(FRACTIONS) - 1):
        high, low = FRACTIONS[i:i + 2]
        if high >= fraction >= low:
            t = (fraction - low) / (high - low)
            return values[i + 1] * (1 - t) + values[i] * t
    return values[-1]


def features(hole, board, value=None):
    """Return made-hand quality, draw quality, and board wetness heuristics."""
    if value is None:
        value = evaluate_sum(sum(ENC[c] for c in hole + board))
    category, primary = value >> 20, (value >> 16) & 15
    hr = [c // 4 + 2 for c in hole]
    br = [c // 4 + 2 for c in board]
    top = max(br)
    paired = len(set(br)) < len(br)
    hole_matches = sum(r in br for r in hr)
    if category >= 4:
        made = {4: .86, 5: .93, 6: .97, 7: .995, 8: 1.0}[category]
        if len(board) == 5 and value == evaluate_sum(sum(ENC[c] for c in board)):
            made = .10
    elif category == 3:
        made = .84 if primary in hr else .22
    elif category == 2:
        made = .76 if not paired else (.59 if hole_matches or hr[0] == hr[1] else .18)
    elif category == 1:
        if hr[0] == hr[1]:
            made = .67 if hr[0] > top else (.44 if hr[0] >= sorted(set(br))[-min(2, len(set(br)))] else .29)
        elif primary in hr:
            made = .56 + .04 * (max(hr) >= 12) if primary == top else .33
        else:
            made = .12 + .025 * sum(r > top for r in hr)
    else:
        made = .08 + .03 * sum(r > top for r in hr) + .025 * (14 in hr)
    suit_counts = [sum(c % 4 == s for c in board) for s in range(4)]
    wet = min(1.0, .18 * max(suit_counts) + .12 * sum(b - a <= 2 for a, b in zip(sorted(set(br)), sorted(set(br))[1:])))
    draw = 0.0
    if len(board) < 5:
        allcards = hole + board
        for s in range(4):
            suited_hole = [c // 4 + 2 for c in hole if c % 4 == s]
            count = suit_counts[s] + len(suited_hole)
            if count == 4 and suited_hole:
                draw = max(draw, .36 if max(suited_hole) == 14 else .29)
            elif count == 3 and len(board) == 3 and suited_hole:
                draw = max(draw, .06)
        rankset = set(hr + br)
        if 14 in rankset:
            rankset.add(1)
        outs = set()
        for low in range(1, 11):
            run = set(range(low, low + 5))
            if len(run & rankset) == 4 and any(r in run for r in hr):
                outs.update(run - rankset)
        draw = min(.55, draw + (.25 if len(outs) >= 2 else .14 if outs else 0.0))
    return made, draw, wet


def equity(hole, board, opponents, rng, comparisons=48, deadline_ns=None, pot_groups=None):
    """opponents: sequence of (preflop range fraction, current aggression).

    Sample complete disjoint deals. Importance weights model stronger ranges
    after postflop betting. Ties split the pot. The cap makes timing predictable;
    these are estimates, not exact equity or a solved equilibrium strategy.
    """
    n = len(opponents)
    if n == 0:
        return 1.0
    known = hole + board
    dead = sum(1 << c for c in known)
    base_board = sum(ENC[c] for c in board)
    hero_sum = ENC[hole[0]] + ENC[hole[1]]
    remaining = 5 - len(board)
    samples = max(8, comparisons // n)
    ranges = []
    pools = {}
    for fraction, aggression in opponents:
        limit = max(24, min(1326, int(fraction * 1326)))
        if limit not in pools:
            pools[limit] = tuple(i for i in range(limit) if not COMBO_MASK[i] & dead) if limit <= 132 else None
        ranges.append((limit, aggression, pools[limit]))
    group_total = sum(amount for amount, _ in pot_groups) if pot_groups else 0
    result = total_weight = 0.0
    random_bits = rng.getrandbits
    enc, masks, sums, evaluate = ENC, COMBO_MASK, COMBO_SUM, evaluate_sum
    board_top = max(c // 4 + 2 for c in board)
    board_pairs = len(set(c // 4 for c in board)) < len(board)
    board_rank_mask = 0
    board_suits = [0] * 4
    for c in board:
        board_rank_mask |= 1 << (c // 4)
        board_suits[c % 4] += 1
    draw_suits_mask = sum(4 << (4 * s) for s in range(4) if board_suits[s] < 4)
    completed = 0
    for sample in range(samples):
        if sample and sample % 2 == 0 and deadline_ns is not None and perf_counter_ns() >= deadline_ns:
            break
        used = dead
        opp_hands = []
        weight = 1.0
        for limit, aggression, pool in ranges:
            for collision in range(12):
                # 32-bit modular reduction has at most 1/2**32 absolute
                # probability bias per index and avoids Python randrange loops.
                index = pool[random_bits(32) % len(pool)] if pool else random_bits(32) % limit
                if not masks[index] & used:
                    break
            else:
                # Search the declared range first; broaden only when it is
                # literally exhausted by known and previously sampled cards.
                valid = tuple(j for j in (pool if pool is not None else range(limit)) if not masks[j] & used)
                if not valid:
                    valid = tuple(j for j in range(1326) if not masks[j] & used)
                index = valid[random_bits(32) % len(valid)]
            chosen = sums[index]
            if aggression:
                partial = base_board + chosen
                score = evaluate(partial)
                cat, rank = score >> 20, (score >> 16) & 15
                if cat >= 3:
                    strength = .98
                elif cat == 2:
                    strength = .9 if not board_pairs else .65
                elif cat == 1:
                    strength = .80 if rank >= board_top else (.4 if not board_pairs else .24)
                else:
                    strength = .14
                if remaining and cat < 3:
                    suits = (partial >> 39) & 0xFFFF
                    flush_draw = bool(suits & draw_suits_mask)
                    straight_draw = STRAIGHT_OUTS[board_rank_mask | COMBO_RANKS[index]]
                    if flush_draw or straight_draw >= 2:
                        strength = max(strength, .50 if len(board) == 3 else .38)
                    elif straight_draw:
                        strength = max(strength, .26)
                weight *= (.18 + .82 * strength if aggression == 1 else .06 + .94 * strength * strength)
            used |= masks[index]
            opp_hands.append(chosen)
        runout = base_board
        for _ in range(remaining):
            for collision in range(16):
                c = random_bits(32) % 52
                if not used & (1 << c):
                    break
            else:
                c = next(j for j in range(52) if not used & (1 << j))
            used |= 1 << c
            runout += enc[c]
        hs = evaluate(hero_sum + runout)
        if pot_groups:
            scores = [evaluate(hand + runout) for hand in opp_hands]
            won = 0.0
            for amount, eligible in pot_groups:
                ties = 1
                for i in eligible:
                    if scores[i] > hs:
                        break
                    ties += scores[i] == hs
                else:
                    won += amount / ties
            result += weight * won / group_total
        else:
            ties = 1
            for hand in opp_hands:
                score = evaluate(hand + runout)
                if score > hs:
                    break
                if score == hs:
                    ties += 1
            else:
                result += weight / ties
        total_weight += weight
        completed += 1
    # Small shrinkage limits overconfidence from a short, bounded sample.
    prior = total_weight / completed * .35
    prior_equity = (sum(amount / (len(eligible) + 1) for amount, eligible in pot_groups) / group_total
                    if pot_groups else 1 / (n + 1))
    return (result + prior * prior_equity) / (total_weight + prior)
