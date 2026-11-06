"""Original, dependency-free Hold'em evaluation and offline equity inference.

Cards are integers ``4*(rank-2)+suit``; suits c,d,h,s. Scores increase
with hand strength, category is ``score >> 20``. The optional compact model
is trained offline by training/generate.cpp and training/train_model.py.
"""
from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib

_BASE = Path(__file__).resolve().parent
USE_EXACT_RIVER = True
_STRAIGHT = [0] * 8192
_TOP = [0] * 8192
for _m in range(8192):
    _s = 0
    for _r in range(12, 3, -1):
        if (_m >> (_r - 4)) & 31 == 31:
            _s = _r + 2
            break
    if not _s and _m & 4111 == 4111:
        _s = 5
    _STRAIGHT[_m] = _s
    _rs = [r + 2 for r in range(12, -1, -1) if _m & (1 << r)][:5]
    _TOP[_m] = sum(r << (16 - 4 * i) for i, r in enumerate(_rs))


def evaluate(cards):
    """Exact best five-card rank for a collection of 5, 6 or 7 cards."""
    counts = [0] * 13
    suits = [0, 0, 0, 0]
    mask = 0
    for c in cards:
        r = c >> 2
        bit = 1 << r
        counts[r] += 1
        suits[c & 3] |= bit
        mask |= bit
    flush = 0
    for s in suits:
        if s.bit_count() >= 5:
            sf = _STRAIGHT[s]
            if sf:
                return (8 << 20) | (sf << 16)
            flush = _TOP[s]
            break
    trips = []
    pairs = []
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 4:
            k = (mask ^ (1 << r)).bit_length() + 1
            return (7 << 20) | ((r + 2) << 16) | (k << 12)
        if n >= 3:
            trips.append(r)
        if n >= 2:
            pairs.append(r)
    if trips and len(pairs) >= 2:
        t = trips[0]
        p = pairs[0] if pairs[0] != t else pairs[1]
        return (6 << 20) | ((t + 2) << 16) | ((p + 2) << 12)
    if flush:
        return (5 << 20) | flush
    st = _STRAIGHT[mask]
    if st:
        return (4 << 20) | (st << 16)
    if trips:
        t = trips[0]
        kick = _TOP[mask ^ (1 << t)]
        return (3 << 20) | ((t + 2) << 16) | ((kick >> 4) & 65280)
    if len(pairs) >= 2:
        p, q = pairs[:2]
        k = (mask ^ (1 << p) ^ (1 << q)).bit_length() + 1
        return (2 << 20) | ((p + 2) << 16) | ((q + 2) << 12) | (k << 8)
    if pairs:
        p = pairs[0]
        kick = _TOP[mask ^ (1 << p)]
        return (1 << 20) | ((p + 2) << 16) | ((kick >> 4) & 65520)
    return _TOP[mask]


def _features(hole, board):
    cards = tuple(hole) + tuple(board)
    score = evaluate(cards)
    bscore = evaluate(board)
    h0, h1 = sorted(hole, key=lambda c: (c // 4, sum((b & 3) == (c & 3) for b in board)), reverse=True)
    a, b = h0 // 4 + 2, h1 // 4 + 2
    if b > a:
        a, b = b, a
    counts = [0] * 15
    suits = [0] * 4
    bsuits = [0] * 4
    mask = 0
    bmask = 0
    ranks = []
    for c in board:
        r = c // 4 + 2
        ranks.append(r)
        counts[r] += 1
        suits[c & 3] += 1
        bsuits[c & 3] += 1
        bmask |= 1 << (r - 2)
    for c in hole:
        suits[c & 3] += 1
    mask = bmask | (1 << (a - 2)) | (1 << (b - 2))
    # All distinct four-to-a-straight completion ranks, including the wheel.
    outs = 0
    occupancy = 0
    boccupancy = 0
    hparticipation = 0
    for window in (4111, 31, 62, 124, 248, 496, 992, 1984, 3968, 7936):
        n = (mask & window).bit_count()
        if n > occupancy:
            occupancy = n
            hparticipation = ((mask ^ bmask) & window).bit_count()
        boccupancy = max(boccupancy, (bmask & window).bit_count())
        if n == 4:
            outs |= window & ~mask
    dom = max(range(4), key=lambda s: (suits[s], bsuits[s], max((c // 4 + 2 for c in hole if c % 4 == s), default=0)))
    suited_hole = sorted((c // 4 + 2 for c in hole if (c & 3) == dom), reverse=True)
    flushhigh = suited_hole[0] if suited_hole and suits[dom] >= 3 else 0
    flushlow = suited_hole[1] if len(suited_hole) > 1 else 0
    highboard = max(ranks, default=0)
    sranks = sorted(ranks, reverse=True) + [0] * (5 - len(ranks))
    cat = score >> 20
    bc = bscore >> 20
    fs = [len(board), cat]
    fs += [(score >> sh) & 15 for sh in (16, 12, 8, 4, 0)]
    fs += [bc] + [(bscore >> sh) & 15 for sh in (16, 12, 8, 4, 0)]
    fs += [a, b, int(a == b), int((h0 & 3) == (h1 & 3)), max(suits), max(bsuits),
           suits[h0 & 3], suits[h1 & 3], flushhigh, outs.bit_count(), occupancy,
           int(a > highboard) + int(b > highboard), bmask.bit_count(),
           max(counts), highboard, min(ranks, default=0), counts[h0 // 4 + 2],
           counts[h1 // 4 + 2], hparticipation, boccupancy, flushlow,
           int(flushhigh == 14), cat - bc] + sranks
    return fs


try:
    _MODEL = json.loads((_BASE / "assets" / "equity_model.json").read_text())
except FileNotFoundError:
    _MODEL = None
try:
    _PREFLOP = json.loads((_BASE / "assets" / "preflop.json").read_text())
except FileNotFoundError:
    _PREFLOP = None
try:
    _NARROW = json.loads((_BASE / 'assets' / 'preflop_narrow.json').read_text())
except FileNotFoundError:
    _NARROW = None
try:
    from assets.equity_predict import predict as _FAST_PREDICT
except ImportError:
    _FAST_PREDICT = None
try:
    _RANK_SCORES = dict(struct.iter_unpack('<II',zlib.decompress((_BASE / 'assets' / 'rank_scores.zlib').read_bytes())))
except FileNotFoundError:
    _RANK_SCORES = None
_RANK_POWERS = [5 ** r for r in range(13)]
_RANK_PAIRS = []
_SUIT_TIGHT = [[False]*13 for _ in range(13)]
_OFF_TIGHT = [[False]*13 for _ in range(13)]
if _PREFLOP is not None:
    _order = _PREFLOP['order']
    for _a in range(13):
        for _b in range(_a,13):
            _RANK_PAIRS.append((_a,_b,_RANK_POWERS[_a]+_RANK_POWERS[_b],
                                _order[13*_b+_a]<266,_order[13*_a+_b]<266))
            _SUIT_TIGHT[_a][_b] = _SUIT_TIGHT[_b][_a] = _order[13*_b+_a]<266
            _OFF_TIGHT[_a][_b] = _OFF_TIGHT[_b][_a] = _order[13*_a+_b]<266


def exact_dry_river(hole,board):
    """Exact HU equity on rivers with at most two board cards of any suit.

    Such a board cannot make a flush with two private cards. All 990 legal
    opponent combinations collapse to 91 rank pairs, with blockers counted
    exactly and the fixed preflop range split by suited/offsuit class.
    Returns ``None`` for other boards or missing optional table assets.
    This function is initially isolated from production inference so its
    exact-enumeration tests and speed can be assessed independently.
    """
    if len(board)!=5 or _RANK_SCORES is None or not _RANK_PAIRS:return None
    suits=[0,0,0,0]
    for c in board:suits[c&3]+=1
    if max(suits)>2:return None
    available=[15]*13
    boardkey=0
    for c in board:
        boardkey+=_RANK_POWERS[c>>2]
        available[c>>2]^=1<<(c&3)
    for c in hole:available[c>>2]^=1<<(c&3)
    nr=[m.bit_count() for m in available]
    score=_RANK_SCORES[boardkey+_RANK_POWERS[hole[0]>>2]+_RANK_POWERS[hole[1]>>2]]
    wins=0;twins=0;nt=0
    for r,s,key,suittight,offtight in _RANK_PAIRS:
        n=nr[r]*(nr[r]-1)//2 if r==s else nr[r]*nr[s]
        if not n:continue
        if r==s:tn=n if suittight else 0
        elif suittight==offtight:tn=n if suittight else 0
        else:
            suited=(available[r]&available[s]).bit_count()
            tn=suited if suittight else n-suited
        os=_RANK_SCORES[boardkey+key]
        w=2 if score>os else 1 if score==os else 0
        wins+=n*w;twins+=tn*w;nt+=tn
    return wins/1980, twins/(2*nt)


def exact_river(hole,board,with_ties=False,_hero_score=None):
    """Exact legal-combination HU river equity, including all flush textures.

    Rank aggregation handles nonflush outcomes. Only combinations completing
    a flush in a board suit with >=3 cards need a correction. Matching-suit
    pairs count individually; one matching card plus an off-suit card can be
    grouped by the latter's rank. At most 91+153 small cases replace all 990
    full hand evaluations. Uses only an original 143KB rank table.
    """
    if len(board)!=5 or _RANK_SCORES is None or not _RANK_PAIRS:return None
    powers=_RANK_POWERS;rscores=_RANK_SCORES
    available=[15]*13;suits=[0,0,0,0];boardmasks=[0,0,0,0];boardkey=0
    for c in board:
        r=c>>2;s=c&3
        boardkey+=powers[r];available[r]^=1<<s;suits[s]+=1;boardmasks[s]|=1<<r
    for c in hole:available[c>>2]^=1<<(c&3)
    nr=[m.bit_count() for m in available]
    ds=max(range(4),key=suits.__getitem__);ns=suits[ds]
    hscore=_hero_score if _hero_score is not None else (evaluate(tuple(hole)+tuple(board)) if ns>=3 else rscores[boardkey+powers[hole[0]>>2]+powers[hole[1]>>2]])
    bflush=evaluate(board) if ns==5 else 0
    wins=twins=nt=ties=tties=0
    for r,s,key,suittight,offtight in _RANK_PAIRS:
        n=nr[r]*(nr[r]-1)//2 if r==s else nr[r]*nr[s]
        if not n:continue
        if r==s:tn=n if suittight else 0
        elif suittight==offtight:tn=n if suittight else 0
        else:
            suited=(available[r]&available[s]).bit_count()
            tn=suited if suittight else n-suited
        os=rscores[boardkey+key]
        if bflush>os:os=bflush
        w=2 if hscore>os else 1 if hscore==os else 0
        wins+=n*w;twins+=tn*w;nt+=tn
        if w==1:ties+=n;tties+=tn
    if ns>=3:
        dm=1<<ds;bm=boardmasks[ds]
        matching=[r for r in range(13) if available[r]&dm]
        # Two cards of the dominant board suit. These are always suited
        # opponent classes, regardless of the board's other suits.
        for i,r in enumerate(matching):
            pr=powers[r];mr=bm|(1<<r)
            for s in matching[i+1:]:
                mask=mr|(1<<s)
                sh=_STRAIGHT[mask]
                fs=(8<<20)|(sh<<16) if sh else (5<<20)|_TOP[mask]
                os=rscores[boardkey+pr+powers[s]]
                if bflush>os:os=bflush
                if fs<=os:continue
                old=2 if hscore>os else 1 if hscore==os else 0
                new=2 if hscore>fs else 1 if hscore==fs else 0
                delta=new-old;wins+=delta
                if _SUIT_TIGHT[r][s]:twins+=delta
                td=int(new==1)-int(old==1);ties+=td
                if _SUIT_TIGHT[r][s]:tties+=td
        if ns>=4:
            # Exactly one matching-suit card; the other card has at most
            # three suits per rank and is aggregated with exact blockers.
            off=[(available[s]&~dm).bit_count() for s in range(13)]
            for r in matching:
                mask=bm|(1<<r);sh=_STRAIGHT[mask]
                fs=(8<<20)|(sh<<16) if sh else (5<<20)|_TOP[mask]
                pr=powers[r]
                for s,n in enumerate(off):
                    if not n:continue
                    os=rscores[boardkey+pr+powers[s]]
                    if bflush>os:os=bflush
                    if fs<=os:continue
                    old=2 if hscore>os else 1 if hscore==os else 0
                    new=2 if hscore>fs else 1 if hscore==fs else 0
                    delta=n*(new-old);wins+=delta
                    if _OFF_TIGHT[r][s]:twins+=delta
                    td=n*(int(new==1)-int(old==1));ties+=td
                    if _OFF_TIGHT[r][s]:tties+=td
    if with_ties:return wins/1980,twins/(2*nt),ties/990,tties/nt
    return wins/1980,twins/(2*nt)


def hand_key(hole):
    a, b = hole[0] // 4, hole[1] // 4
    if b > a:
        a, b = b, a
    if a == b:
        return 13 * a + b
    return 13 * (a if hole[0] % 4 == hole[1] % 4 else b) + (b if hole[0] % 4 == hole[1] % 4 else a)


def preflop_equities(hole):
    """HU equities against top 5/10/20/35/50/100% preflop ranges."""
    if _PREFLOP is not None:
        return _PREFLOP["equities"][hand_key(hole)]
    a, b = sorted((c // 4 + 2 for c in hole), reverse=True)
    q = min(.86, .25 + .023 * a + .007 * b + (.24 if a == b else 0) + (.025 if hole[0] % 4 == hole[1] % 4 else 0))
    return [max(.1, q - x) for x in (.2, .16, .11, .07, .04, 0)]


def preflop_narrow_equities(hole):
    """HU equities against nominal top 1%/2.5% ranges (whole hand classes).

    This separate API preserves the original six-range return convention.
    Actual combo fractions and MC settings are recorded in the narrow asset.
    """
    if _NARROW is not None:return _NARROW['equities'][hand_key(hole)]
    e=preflop_equities(hole)[0]
    return [max(.02,e-.05),max(.02,e-.025)]


def _infer(fs):
    if _FAST_PREDICT is not None:
        er, et = _FAST_PREDICT(fs)
        return max(0.,min(1.,er)), max(0.,min(1.,et))
    if _MODEL is None:
        # Startup-safe conservative heuristic, replaced by shipped trained model.
        cat = fs[1]
        q = (.22, .58, .77, .87, .92, .95, .98, .995, .999)[cat]
        q += .006 * (fs[2] - 8)
        if fs[0] < 5:
            q += .025 * fs[22] + (.08 if fs[17] == 4 else 0)
        return min(.999, max(.01, q)), min(.999, max(.01, q - .10))
    ys = list(_MODEL["base"])
    for tree in _MODEL["trees"]:
        f, threshold, left, right, values = tree
        node = 0
        while left[node] >= 0:
            node = left[node] if fs[f[node]] <= threshold[node] else right[node]
        v = values[node]
        ys[0] += v[0]
        ys[1] += v[1]
    return max(0., min(1., ys[0])), max(0., min(1., ys[1]))


def analyze(hole, board):
    """Return fast showdown equity estimates plus strategic hand features.

    ``equity_tight`` uses a fixed top-20%-by-preflop-strength range; it is
    not a solved posterior against an opponent's particular betting line.
    """
    if not board:
        es = preflop_equities(hole)
        return {"equity_random": es[-1], "equity_tight": es[2], "category": int(hole[0] // 4 == hole[1] // 4),
                "draw_strength": 0., "nut_potential": 0., "score": 0}
    fs = _features(hole, board)
    score = (fs[1] << 20) | sum(fs[i + 2] << sh for i, sh in enumerate((16,12,8,4,0)))
    exact = exact_river(hole,board,with_ties=True,_hero_score=score) if USE_EXACT_RIVER and len(board)==5 else None
    tier = tiet = None
    if exact is None:
        er, et = _infer(fs)
    else:
        er,et,tier,tiet=exact
    # Three analytically locked river boards; these are rare enough that a
    # learned approximation should not be trusted to discover exact ties.
    locked = False
    if len(board) == 5:
        bc, high, kick = fs[7], fs[8], fs[9]
        locked = ((bc == 8 and high == 14) or
                  (bc == 7 and kick == (13 if high == 14 else 14)) or
                  (bc == 4 and high == 14 and fs[18] < 3))
        if locked:
            er = et = .5
    fd = fs[17] == 4 and fs[21] > 0 and len(board) < 5
    sd = fs[22] * (8 / 47 if len(board) == 3 else 4 / 46) if len(board) < 5 and fs[31] else 0.
    draw = min(.55, (.34 if len(board) == 3 else .196) * fd + sd)
    return {"equity_random": er, "equity_tight": et, "category": fs[1],
            "draw_strength": draw, "nut_potential": float(fs[1] >= 4 or (fd and fs[34]) or fs[1] == 3),
            "score": score,
            "board_category": fs[7], "flush_draw": bool(fd), "straight_outs": fs[22],
            "overcards": fs[24], "flush_cards": fs[18], "hole_pair_matches": (fs[29], fs[30]),
            "locked_split": locked, "tie_random": tier, "tie_tight": tiet,
            "exact_river": exact is not None}


def equity(hole, board, tightness=0.):
    a = analyze(hole, board)
    t = max(0., min(1., tightness))
    return a["equity_random"] * (1 - t) + a["equity_tight"] * t
