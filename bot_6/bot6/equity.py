"""Offline range assets and bounded, weighted equity sampling."""
import json
from time import perf_counter_ns
from pathlib import Path
from .cards import CARD_MASK, evaluate_mask, hand_class, HIGH

with (Path(__file__).resolve().parent.parent / "assets" / "preflop.json").open() as _f:
    DATA = json.load(_f)
WIDTHS = DATA["widths"]
ORDER = {name: i for i, name in enumerate(DATA["order"])}
COMBOS = sorted(((a, b, CARD_MASK[a] | CARD_MASK[b], hand_class(a, b))
                 for a in range(52) for b in range(a)), key=lambda x: ORDER[x[3]])
def _whole_class_range(width):
    count = max(6, round(1326 * width))
    while count < len(COMBOS) and COMBOS[count][3] == COMBOS[count-1][3]:
        count += 1
    return tuple(COMBOS[:count])


RANGES = tuple(_whole_class_range(w) for w in WIDTHS)
PERCENTILE = {}
_n = 0
for _name in DATA["order"]:
    _count = 6 if len(_name) == 2 else 4 if _name[-1] == "s" else 12
    PERCENTILE[_name] = (_n + _count / 2) / 1326
    _n += _count


def range_index(width):
    return min(range(len(WIDTHS)), key=lambda i: abs(WIDTHS[i] - width))


def preflop_equity(hole, width):
    values = DATA["equity"][hand_class(*hole)]
    if width >= WIDTHS[0]:
        return values[0]
    for i in range(1, len(WIDTHS)):
        if width >= WIDTHS[i]:
            fraction = (width - WIDTHS[i]) / (WIDTHS[i-1] - WIDTHS[i])
            return values[i] + fraction * (values[i-1] - values[i])
    return values[-1]


def texture(board):
    ranks = [c // 4 + 2 for c in board]
    suits = [sum(c % 4 == s for c in board) for s in range(4)]
    mask = 0
    for r in ranks:
        mask |= 1 << (r - 2)
    connected = max(((mask >> i) & 31).bit_count() for i in range(9))
    wet = min(1., .24 * max(0, max(suits, default=0) - 1) + .18 * max(0, connected - 1))
    return mask, max(ranks, default=2), wet


def hand_quality(a, b, board_mask, rank_mask, board_high, nboard):
    """A likelihood feature, not an equity estimate."""
    score = evaluate_mask(board_mask | CARD_MASK[a] | CARD_MASK[b])
    cat = score >> 24
    r = (score >> 20) & 15
    ra, rb = a // 4 + 2, b // 4 + 2
    if cat == 0:
        made = .04 + .035 * ((ra > board_high) + (rb > board_high))
    elif cat == 1:
        own_pair = ra == r or rb == r
        made = (.69 if r > board_high else .59 if r == board_high else .37 if r >= board_high - 3 else .24) if own_pair else .10 + .025 * (max(ra, rb) > board_high)
    elif cat == 2:
        lowpair = (score >> 16) & 15
        made = .77 if ra in (r, lowpair) and rb in (r, lowpair) and ra != rb else .62 if ra in (r, lowpair) or rb in (r, lowpair) else .24
    elif cat == 3:
        made = .88 if ra == r or rb == r else .31
    else:
        made = (.91, .94, .98, .995, 1.)[cat - 4]
    draw = 0.
    if nboard < 5:
        mask = board_mask | CARD_MASK[a] | CARD_MASK[b]
        for s in {a % 4, b % 4}:
            if ((mask >> (13 * s)) & 8191).bit_count() == 4:
                draw = .20
        ranks = rank_mask | (1 << (ra - 2)) | (1 << (rb - 2))
        # Include A2345 as well as ordinary straight windows.
        if any(((ranks >> i) & 31).bit_count() == 4 for i in range(9)) or (ranks & 4111).bit_count() == 4:
            draw += .11
    return made, min(.28, draw)


def likelihood(made, draw, pressure):
    # Positive pressure means a bet/raise; negative means a call. Mixture
    # floors preserve plausible bluffs and prevent zero-weight worlds.
    if pressure >= 2:
        return .035 + .965 * min(1., made * made * made * 1.30 + draw * .9)
    if pressure > 0:
        return .12 + .88 * min(1., made * made * 1.12 + draw * 1.3)
    if pressure < 0:
        return .22 + .78 * min(1., made * 1.12 + draw * 1.5)
    return 1.


def estimate(hole, board, specs, rng, budget=120, pots=None, deadline_ns=0):
    """Return (weighted showdown equity, effective samples).

    Each range spec is (preflop width, current-street pressure). Sampling is
    joint without replacement; max attempt counts bound even tiny ranges.
    No per-call allocations proportional to history or the full deck.
    """
    n = len(specs)
    if not n:
        return 1., 1.
    hero = CARD_MASK[hole[0]] | CARD_MASK[hole[1]]
    bm = sum(CARD_MASK[c] for c in board)
    known = hero | bm
    rm, bh, _ = texture(board)
    ranges = [(RANGES[range_index(w)], pressure) for w, pressure in specs]
    # Include likelihood evals in the work budget. Heads up generally uses
    # 30–46 worlds; eight-way uses 6–10. Final latency is benchmarked.
    work = 1 + n + sum(p != 0 for _, p in ranges)
    samples = max(6, budget // work)
    missing = 5 - len(board)
    total = total2 = wins = 0.
    getbits, ev, masks = rng.getrandbits, evaluate_mask, CARD_MASK
    for _ in range(samples):
        if deadline_ns and _ >= 4 and _ % 4 == 0 and perf_counter_ns() >= deadline_ns:
            break
        used, weight = known, 1.
        opponents = []
        for combos, pressure in ranges:
            size = len(combos)
            found = False
            for attempt in range(12):
                a, b, cm, _name = combos[getbits(32) % size]
                if not cm & used:
                    found = True
                    break
            if not found:
                # Bounded uniform fallback: scan a random cyclic deck.
                start = getbits(32) % 52
                a = next((start + k) % 52 for k in range(52) if not masks[(start + k) % 52] & used)
                used |= masks[a]
                b = next((start + k) % 52 for k in range(52) if not masks[(start + k) % 52] & used)
                cm = masks[a] | masks[b]
            used |= cm
            opponents.append(cm)
            if pressure:
                made, draw = hand_quality(a, b, bm, rm, bh, len(board))
                weight *= likelihood(made, draw, pressure)
        runout = bm
        for _j in range(missing):
            start = getbits(32) % 52
            # Linear probing only after a bounded rejection phase.
            for _attempt in range(12):
                if not masks[start] & used:
                    break
                start = getbits(32) % 52
            else:
                while masks[start] & used:
                    start = (start + 1) % 52
            used |= masks[start]
            runout |= masks[start]
        hs = ev(runout | hero)
        if pots:
            scores = [ev(runout | cm) for cm in opponents]
            share, pot_total = 0., 0
            for amount, eligible in pots:
                tied, beaten = 1, False
                for i, vs in enumerate(scores):
                    if eligible & (1 << i):
                        if vs > hs:
                            beaten = True
                            break
                        tied += vs == hs
                if not beaten:
                    share += amount / tied
                pot_total += amount
            share /= max(1, pot_total)
        else:
            tied, beaten = 1, False
            for cm in opponents:
                vs = ev(runout | cm)
                if vs > hs:
                    beaten = True
                    break
                tied += vs == hs
            share = 0. if beaten else 1. / tied
        total += weight
        total2 += weight * weight
        wins += weight * share
    return wins / total, total * total / total2
