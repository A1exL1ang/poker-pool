"""Bounded-work equity estimation; all assets are loaded before decisions."""
import json
import ctypes
from pathlib import Path
from .evaluator import BITS, TOP1, STRAIGHT, evaluate_bits, packed
from .native import LIB

with (Path(__file__).resolve().parent.parent / "assets" / "preflop.json").open() as _f:
    TABLE = json.load(_f)
FRACTIONS = TABLE["fractions"]
COMBOS = tuple((a, b, BITS[a] | BITS[b]) for a, b in TABLE["combos"])
if LIB is not None:
    LIB.bot_init((ctypes.c_uint64 * 1326)(*(c[2] for c in COMBOS)))
STRENGTH = TABLE["strength"]
VERSUS = TABLE["versus_range"]
MULTIWAY = TABLE["multiway_random"]


def hand_class(a, b):
    x, y = a % 13, b % 13
    h, l = max(x, y), min(x, y)
    return h * 13 + l if a // 13 == b // 13 else l * 13 + h


def _open_score(k):
    a, b = divmod(k, 13)
    h, l = max(a, b), min(a, b)
    suited = a > b
    score = STRENGTH[k]
    if suited:
        score += .028
        if h - l <= 2:
            score += .012
    elif a != b and l < 7:
        score -= .014
    if h == 12 and suited and l <= 3:
        score += .009
    return score


OPEN_PERCENTILE = [0.0] * 169
_total = 0
for _k in sorted(range(169), key=_open_score, reverse=True):
    _a, _b = divmod(_k, 13)
    _n = 6 if _a == _b else 4 if _a > _b else 12
    OPEN_PERCENTILE[_k] = (_total + _n / 2) / 1326
    _total += _n


def preflop_equity(k, fraction, opponents=1):
    fraction = max(.05, min(1., fraction))
    row = VERSUS[k]
    eq = row[-1]
    for i, f in enumerate(FRACTIONS):
        if fraction <= f:
            if i == 0:
                eq = row[0]
            else:
                t = (fraction - FRACTIONS[i-1]) / (f - FRACTIONS[i-1])
                eq = row[i-1] * (1-t) + row[i] * t
            break
    if opponents > 1:
        # Calibrate the independent-opponent approximation to directly simulated
        # multiway random equity, preserving pair/connector playability.
        random_hu = MULTIWAY[k][0]
        exponent = 1.0
        if random_hu > 0 and random_hu < 1:
            import math
            exponent = math.log(max(.001, MULTIWAY[k][min(7, opponents)-1])) / math.log(random_hu)
        eq = eq ** exponent
    return eq


def draw_quality(hole, board):
    """Draw potential on an incomplete board, in [0,1]; no fake river outs."""
    if len(board) >= 5:
        return 0.0
    cards = hole + board
    ranks = 0
    suits = [0, 0, 0, 0]
    for c in cards:
        ranks |= 1 << (c % 13)
        suits[c // 13] += 1
    fd = max(suits) == 4 and any(suits[c // 13] == 4 for c in hole)
    straight_outs = 0
    if not STRAIGHT[ranks]:
        for r in range(13):
            if not ranks & (1 << r) and STRAIGHT[ranks | (1 << r)]:
                straight_outs += 1
    # An out completing a board-only straight is deliberately discounted.
    board_ranks = 0
    for c in board:
        board_ranks |= 1 << (c % 13)
    connected = any(not board_ranks & (1 << (c % 13)) for c in hole)
    return min(1., .62 * fd + .25 * straight_outs * connected)


def board_texture(board):
    ranks = sorted(set(c % 13 for c in board))
    suits = [sum(c // 13 == s for c in board) for s in range(4)]
    wet = max(suits) >= 2 and len(board) == 3 or max(suits) >= 3
    close = max((sum(0 <= r-x <= 4 for r in ranks) for x in ranks), default=0) >= 3
    return min(1., .45 * wet + .4 * close + .15 * (len(ranks) < len(board)))


class EquityEstimator:
    """Small stratified Monte Carlo with blocker removal and action weights.

    The uncertainty estimate is diagnostic, not a statistical confidence bound:
    samples are stratified and opponents' ranges are modeled heuristically.
    """
    def __init__(self, seed=0xB071):
        self.seed = seed & 0xffffffff or 1

    def estimate(self, hole, board, fractions, aggression=0., trials=None, layers=None):
        if LIB is not None:
            n = len(fractions)
            if n == 0:
                return 1., 0.
            if n > 7:
                raise ValueError("at most seven opponents")
            limits = (ctypes.c_int * n)(*(max(66,min(1326,int(f*1326))) for f in fractions))
            if len(board) == 5 and n == 1 and trials is None:
                value=LIB.bot_river(packed(hole),packed(board),limits[0],aggression)
                if layers:
                    value=sum(w*(value if mask else 1.) for mask,w in layers)
                return value,0.
            count=len(layers) if layers else 0
            masks=(ctypes.c_int*count)(*(m for m,w in layers)) if layers else None
            weights=(ctypes.c_double*count)(*(w for m,w in layers)) if layers else None
            state = ctypes.c_uint32(self.seed)
            error = ctypes.c_double()
            value = LIB.bot_estimate(packed(hole),packed(board),5-len(board),limits,n,trials or max(32,320//(n+2)),aggression,ctypes.byref(state),ctypes.byref(error),masks,weights,count)
            self.seed = state.value
            return value,error.value
        return self.estimate_python(hole,board,fractions,aggression,trials,layers)

    def estimate_python(self, hole, board, fractions, aggression=0., trials=None, layers=None):
        n = len(fractions)
        if n == 0:
            return 1., 0.
        # A hard evaluator-work cap keeps even 8-way decisions bounded.
        trials = trials or max(8, 72 // (n + 2))
        known = packed(hole + board)
        hero = packed(hole)
        boardbits = packed(board)
        missing = 5 - len(board)
        limits = [max(66, min(1326, int(f * 1326))) for f in fractions]
        seed = self.seed ^ (known & 0xffffffff) ^ (known >> 26)
        win_sum = weight_sum = square_sum = 0.
        ev = evaluate_bits
        combos = COMBOS
        bits = BITS
        # Stratification across the entire prior range suppresses the most
        # expensive source of variance without a runtime range-construction pass.
        for t in range(trials):
            used = known
            villains = []
            for j, lim in enumerate(limits):
                seed = (1664525 * seed + 1013904223) & 0xffffffff
                idx = ((t * lim // trials) + ((seed >> 16) % max(1, lim // trials)) + j * 317) % lim
                a, b, vb = combos[idx]
                tries = 0
                while vb & used:
                    seed = (1664525 * seed + 1013904223) & 0xffffffff
                    a, b, vb = combos[(seed >> 8) % lim]
                    tries += 1
                    if tries == 20:
                        # Narrow top ranges can be exhausted by blockers at
                        # many-way tables; widen instead of looping forever.
                        lim = 1326
                villains.append(vb)
                used |= vb
            runout = boardbits
            for _ in range(missing):
                seed = (1664525 * seed + 1013904223) & 0xffffffff
                c = (seed >> 8) % 52
                while used & bits[c]:
                    seed = (1664525 * seed + 1013904223) & 0xffffffff
                    c = (seed >> 8) % 52
                used |= bits[c]
                runout |= bits[c]
            score = ev(runout | hero)
            ties = 1
            beats_mask = ties_mask = 0
            lost = False
            weight = 1.
            for j, vb in enumerate(villains):
                vscore = ev(runout | vb)
                if vscore > score:
                    lost = True
                    beats_mask |= 1 << j
                elif vscore == score:
                    ties += 1
                    ties_mask |= 1 << j
                if aggression > 0. and j == 0:
                    # Condition on current, observable made strength; never on
                    # future cards. Keep a bluff/draw floor to avoid overfolding.
                    current = ev(boardbits | vb)
                    cat = current >> 24
                    pair_rank = (current >> 12) & 15
                    top_board = TOP1[(boardbits & 8191) | ((boardbits >> 13) & 8191) | ((boardbits >> 26) & 8191) | (boardbits >> 39)]
                    likelihood = .30 if cat == 0 else .62 if cat == 1 and pair_rank < top_board else 1.15 if cat == 1 else 1.8 if cat == 2 else 2.5
                    if cat == 0 and missing:
                        cb = boardbits | vb
                        if ((cb & 8191).bit_count() == 4 or ((cb >> 13) & 8191).bit_count() == 4 or ((cb >> 26) & 8191).bit_count() == 4 or (cb >> 39).bit_count() == 4):
                            likelihood = .9
                    weight *= 1. + aggression * (likelihood - 1.)
            share = 0. if lost else 1. / ties
            if layers:
                share=sum(w/(1+(ties_mask&mask).bit_count()) for mask,w in layers if not beats_mask&mask)
            win_sum += weight * share
            weight_sum += weight
            square_sum += weight * weight
        self.seed = seed
        eq = win_sum / weight_sum
        ess = weight_sum * weight_sum / square_sum
        # Jeffreys-style smoothing guards against false zero/one equities from
        # the bounded sample. River board-only nuts are handled by the policy.
        eq = (eq * ess + .5) / (ess + 1.)
        error = (eq * (1. - eq) / max(1., ess)) ** .5
        return eq, error
