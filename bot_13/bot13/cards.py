"""Small independent card utilities and a bounded Python fallback evaluator.

Card encoding: 4 * (rank - 2) + suit, suit order clubs/diamonds/hearts/spades.
Scores use the same lexicographic category/rank layout as the native kernel.
"""

from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card(value):
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 52:
        return value
    if isinstance(value, str) and len(value) == 2:
        return 4 * RANKS.index(value[0].upper()) + SUITS.index(value[1].lower())
    raise ValueError("a card must be an integer 0..51 or a string such as As")


def text(value):
    return RANKS[value // 4] + SUITS[value % 4]


def hand_class(a, b):
    ra, rb = a // 4, b // 4
    if ra < rb:
        ra, rb = rb, ra
    return RANKS[ra] + RANKS[rb] + ("" if ra == rb else "s" if a % 4 == b % 4 else "o")


def class_index(a, b):
    """13x13: pairs diagonal, suited upper triangle, offsuit lower triangle."""
    ra, rb = a // 4, b // 4
    hi, lo = max(ra, rb), min(ra, rb)
    return hi * 13 + lo if a % 4 == b % 4 or ra == rb else lo * 13 + hi


def _straight(mask):
    run = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    if run:
        return run.bit_length() + 5
    return 5 if mask & 0x100F == 0x100F else 0


def _pack(category, ranks):
    score = category << 24
    for i, rank in enumerate(ranks[:5]):
        score |= rank << (20 - i * 4)
    return score


def evaluate(cards):
    """Evaluate 5..7 unique cards directly; higher scores win."""
    if not 5 <= len(cards) <= 7 or len(set(cards)) != len(cards):
        raise ValueError("evaluate requires 5..7 distinct cards")
    counts = [0] * 13
    suits = [0] * 4
    mask = 0
    for c in cards:
        if not isinstance(c, int) or not 0 <= c < 52:
            raise ValueError("invalid card")
        r = c // 4
        counts[r] += 1
        mask |= 1 << r
        suits[c % 4] |= 1 << r
    flush = next((s for s in suits if s.bit_count() >= 5), 0)
    if flush:
        straight = _straight(flush)
        if straight:
            return _pack(8, [straight])
    quads = [r + 2 for r in range(12, -1, -1) if counts[r] == 4]
    trips = [r + 2 for r in range(12, -1, -1) if counts[r] >= 3]
    pairs = [r + 2 for r in range(12, -1, -1) if counts[r] >= 2]
    ranks = [r + 2 for r in range(12, -1, -1) if counts[r]]
    if quads:
        return _pack(7, quads + [r for r in ranks if r != quads[0]][:1])
    if trips and len(pairs) > 1:
        return _pack(6, [trips[0], next(r for r in pairs if r != trips[0])])
    if flush:
        return _pack(5, [r + 2 for r in range(12, -1, -1) if flush & (1 << r)][:5])
    straight = _straight(mask)
    if straight:
        return _pack(4, [straight])
    if trips:
        return _pack(3, [trips[0]] + [r for r in ranks if r != trips[0]][:2])
    if len(pairs) >= 2:
        return _pack(2, pairs[:2] + [r for r in ranks if r not in pairs[:2]][:1])
    if pairs:
        return _pack(1, pairs[:1] + [r for r in ranks if r != pairs[0]][:3])
    return _pack(0, ranks[:5])


def reference_evaluate(cards):
    """Slow independent five-card decomposition, for validation only."""
    def five(cs):
        ranks = sorted((c // 4 + 2 for c in cs), reverse=True)
        groups = sorted(((ranks.count(r), r) for r in set(ranks)), reverse=True)
        flush = len({c % 4 for c in cs}) == 1
        uniq = sorted(set(ranks), reverse=True)
        straight = (uniq[0] if len(uniq) == 5 and uniq[0] - uniq[-1] == 4
                    else 5 if uniq == [14, 5, 4, 3, 2] else 0)
        if flush and straight:
            return _pack(8, [straight])
        if groups[0][0] == 4:
            return _pack(7, [groups[0][1], groups[1][1]])
        if [g[0] for g in groups] == [3, 2]:
            return _pack(6, [g[1] for g in groups])
        if flush:
            return _pack(5, ranks)
        if straight:
            return _pack(4, [straight])
        category = {3: 3, 2: 1}.get(groups[0][0], 0)
        if len(groups) == 3 and groups[0][0] == groups[1][0] == 2:
            category = 2
        return _pack(category, [g[1] for g in groups])
    return max(five(cs) for cs in combinations(cards, 5))


def features(hole, board):
    """Cheap, board-relative made-hand and draw features; no hidden cards."""
    ranks = [c // 4 + 2 for c in hole]
    br = [c // 4 + 2 for c in board]
    all_cards = tuple(hole) + tuple(board)
    suits = [0] * 4
    masks = 0
    for c in all_cards:
        suits[c % 4] += 1
        masks |= 1 << (c // 4)
    flush_draw = len(board) < 5 and any(
        suits[s] == 4 and any(c % 4 == s for c in hole) for s in range(4))
    straight_outs = 0
    if len(board) < 5 and not _straight(masks):
        present = set(ranks + br)
        board_mask = sum(1 << (r - 2) for r in set(br))
        for r in range(2, 15):
            bit = 1 << (r - 2)
            if r not in present and _straight(masks | bit) > _straight(board_mask | bit):
                straight_outs += 1
    paired = sorted(set(br), reverse=True)
    pair_rank = max((r for r in ranks if r in br), default=0)
    overpair = ranks[0] == ranks[1] and (not br or ranks[0] > max(br))
    top_pair = bool(br and pair_rank == max(br))
    pocket_set = ranks[0] == ranks[1] and ranks[0] in br
    wet = (max((sum(c % 4 == s for c in board) for s in range(4)), default=0) >= 2
           or any(sum(abs(x - y) <= 3 for y in paired) >= 3 for x in paired))
    nut_flush_blocker = any(c // 4 == 12 and sum(b % 4 == c % 4 for b in board) >= 2 for c in hole)
    return {"flush_draw": flush_draw, "straight_draw": straight_outs >= 1,
            "open_ended": straight_outs >= 2, "overpair": overpair,
            "top_pair": top_pair, "pocket_set": pocket_set,
            "pair_rank": pair_rank, "wet": wet,
            "nut_blocker": nut_flush_blocker, "high": max(ranks),
            "kicker": min(ranks)}
