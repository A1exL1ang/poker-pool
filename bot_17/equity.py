"""Offline poker evaluator and bounded equity backend, independently authored.

Ranks: category (0 high card .. 8 straight flush) in bits 20+, followed
by five descending 4-bit rank fields, with actual ranks 2..14.
`range_strength` narrows a heuristic preflop ordering: 0=all combinations,
1=top approximately 5%, with all tied boundary combinations retained.
`aggression` is a separate 0..1 current-board made-hand/draw likelihood tilt.
These are transparent opponent modeling assumptions, not solved ranges.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_NATIVE = None
NATIVE_ERROR = None
if not os.environ.get('POKER_PURE_PYTHON'):
    try:
        _NATIVE = ctypes.CDLL(str(_ROOT / 'native' / ('poker_equity.dll' if os.name == 'nt' else 'poker_equity.so')))
        _NATIVE.poker_init.argtypes = []
        _NATIVE.poker_init.restype = None
        _NATIVE.poker_rank.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        _NATIVE.poker_rank.restype = ctypes.c_uint32
        _NATIVE.poker_equity.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_uint64, ctypes.c_int]
        _NATIVE.poker_equity.restype = ctypes.c_double
        if hasattr(_NATIVE,'poker_equity_pots'):
            _NATIVE.poker_equity_pots.argtypes = _NATIVE.poker_equity.argtypes + [ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int),ctypes.c_int]
            _NATIVE.poker_equity_pots.restype = ctypes.c_double
        if hasattr(_NATIVE,'poker_river_equity'):
            _NATIVE.poker_river_equity.argtypes = [ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.c_double,ctypes.c_double]
            _NATIVE.poker_river_equity.restype = ctypes.c_double
        _NATIVE.poker_init()
    except (OSError, AttributeError) as exc:
        NATIVE_ERROR = str(exc)
        _NATIVE = None
BACKEND = 'native' if _NATIVE is not None else 'python_heuristic'

def _pack(category, ranks):
    result = category << 20
    for i, rank in enumerate(ranks):
        result |= rank << (16 - 4 * i)
    return result

def _straight(mask):
    for high in range(12, 3, -1):
        if mask & (31 << (high - 4)) == 31 << (high - 4):
            return high + 2
    return 5 if mask & 4111 == 4111 else 0

def evaluate_python(cards):
    """Exact 5-7 card rank in pure Python, no external libraries."""
    counts = [0] * 13
    suits = [0] * 4
    suit_counts = [0] * 4
    mask = 0
    for card in cards:
        rank, suit = card >> 2, card & 3
        counts[rank] += 1
        suits[suit] |= 1 << rank
        suit_counts[suit] += 1
        mask |= 1 << rank
    flush = next((s for s in range(4) if suit_counts[s] >= 5), -1)
    if flush >= 0:
        high = _straight(suits[flush])
        if high:
            return _pack(8, [high])
    ranks = [r for r in range(12, -1, -1) if counts[r]]
    quads = [r for r in ranks if counts[r] == 4]
    if quads:
        return _pack(7, [quads[0] + 2, next(r + 2 for r in ranks if r != quads[0])])
    trips = [r for r in ranks if counts[r] == 3]
    pairs = [r for r in ranks if counts[r] == 2]
    if trips and (pairs or len(trips) > 1):
        pair = max(pairs + trips[1:])
        return _pack(6, [trips[0] + 2, pair + 2])
    if flush >= 0:
        return _pack(5, [r + 2 for r in ranks if suits[flush] & (1 << r)][:5])
    high = _straight(mask)
    if high:
        return _pack(4, [high])
    if trips:
        return _pack(3, [trips[0] + 2] + [r + 2 for r in ranks if r != trips[0]][:2])
    if len(pairs) >= 2:
        return _pack(2, [pairs[0] + 2, pairs[1] + 2, next(r + 2 for r in ranks if r not in pairs[:2])])
    if pairs:
        return _pack(1, [pairs[0] + 2] + [r + 2 for r in ranks if r != pairs[0]][:3])
    return _pack(0, [r + 2 for r in ranks[:5]])

def evaluate(cards):
    if not 5 <= len(cards) <= 7:
        raise ValueError('evaluate requires 5 through 7 cards')
    if len(set(cards)) != len(cards) or any(type(c) is not int or c < 0 or c > 51 for c in cards):
        raise ValueError('cards must be distinct integers 0..51')
    if _NATIVE is not None:
        return int(_NATIVE.poker_rank((ctypes.c_int * len(cards))(*cards), len(cards)))
    return evaluate_python(cards)

def category(cards):
    return evaluate(cards) >> 20

def hand_class(hole):
    a, b = hole
    hi, lo = sorted((a >> 2, b >> 2), reverse=True)
    ranks = '23456789TJQKA'
    return ranks[hi] + ranks[lo] + ('' if hi == lo else 's' if a & 3 == b & 3 else 'o')

def starting_score(hole):
    a, b = hole
    hi, lo = sorted((a >> 2, b >> 2), reverse=True)
    if hi == lo:
        return .50 + .0292 * hi
    return .28 + .022 * hi + .009 * lo + (.025 if a & 3 == b & 3 else 0) + (.010 if hi - lo == 1 else 0) + (.015 if hi == 12 else 0)

_COMBO_SCORES = sorted(starting_score((a, b)) for a in range(52) for b in range(a + 1, 52))
_PERCENTILES = {score: sum(s < score for s in _COMBO_SCORES) / 1326 for score in set(_COMBO_SCORES)}

def percentile(hole):
    """Heuristic preflop percentile: zero worst, near one best."""
    return _PERCENTILES[starting_score(hole)]

try:
    with (_ROOT / 'data' / 'preflop.json').open(encoding='utf-8') as handle:
        _PREFLOP = json.load(handle)
except (OSError, ValueError):
    _PREFLOP = None

try:
    with (_ROOT / 'data' / 'equity_model.json').open(encoding='utf-8') as handle:
        _MODEL = json.load(handle)
    if _MODEL.get('format') != 1 or _MODEL.get('features') != 40:
        _MODEL = None
except (OSError, ValueError):
    _MODEL = None
if _NATIVE is None and _MODEL is not None:
    BACKEND = 'python_model'
_COMPILED_PREDICTOR = None
_COMPILED_PREDICTOR_FAST = None
if _MODEL is not None:
    try:
        from data.equity_predictor import predict as _COMPILED_PREDICTOR, predict_fast as _COMPILED_PREDICTOR_FAST, MODEL_SIGNATURE
        if MODEL_SIGNATURE != (_MODEL['features'],len(_MODEL['trees']),_MODEL['base']):
            _COMPILED_PREDICTOR = None
            _COMPILED_PREDICTOR_FAST = None
    except (ImportError, AttributeError):
        _COMPILED_PREDICTOR = None

def _near_straight(mask):
    return max((mask & 4111).bit_count(), max(((mask >> shift) & 31).bit_count() for shift in range(9)))

def _short_board_rank(board):
    if len(board) == 5:
        return evaluate_python(board)
    counts = [0]*13
    for c in board:counts[c >> 2] += 1
    ranks = [r for r in range(12,-1,-1) if counts[r]]
    quads = [r for r in ranks if counts[r] == 4]
    trips = [r for r in ranks if counts[r] == 3]
    pairs = [r for r in ranks if counts[r] == 2]
    if quads:return _pack(7,[quads[0]+2])
    if trips:return _pack(3,[trips[0]+2]+[r+2 for r in ranks if r!=trips[0]][:2])
    if len(pairs)>1:return _pack(2,[r+2 for r in pairs[:2]])
    if pairs:return _pack(1,[pairs[0]+2]+[r+2 for r in ranks if r!=pairs[0]][:3])
    return _pack(0,[r+2 for r in ranks])

def model_features(hole, board, opponents, tightness, aggression):
    """Feature order is mirrored in native/train_fallback.cpp and tested."""
    a,b=hole;nb=len(board);cards=list(hole)+list(board)
    rank=evaluate_python(cards);brank=_short_board_rank(board)
    bc=[0]*13;sc=[0]*4;bm=0;hm=(1<<(a>>2))|(1<<(b>>2))
    for c in board:bc[c>>2]+=1;sc[c&3]+=1;bm|=1<<(c>>2)
    hc=[sc[s]+(a&3==s)+(b&3==s) for s in range(4)]
    hi=max(a>>2,b>>2);lo=min(a>>2,b>>2)
    br=[c>>2 for c in board]
    flushr=sorted((c>>2 for c in hole if hc[c&3]>=4),reverse=True)+[-1,-1]
    return [nb,opponents,tightness,aggression,rank>>20,(rank>>16)&15,(rank>>12)&15,(rank>>8)&15,(rank>>4)&15,rank&15,
        hi,lo,int(hi==lo),int(a&3==b&3),hi-lo,bc[hi],bc[lo],bm.bit_count(),max(br),min(br),
        max(bc),bc.count(2),sum(c>=3 for c in bc),max(sc),max(hc),max(sc[a&3],sc[b&3]),min(sc[a&3],sc[b&3]),flushr[0],flushr[1],
        _near_straight(bm|hm),_near_straight(bm),brank>>20,int(nb==5 and rank==brank),sum(r>hi for r in br),sum(r>lo for r in br),starting_score(hole),
        int(hi==12),(brank>>16)&15,(brank>>12)&15,int(_near_straight(bm|hm)>_near_straight(bm))]

def model_estimate(hole, board, opponents, tightness, aggression):
    if _MODEL is None:
        return _heuristic(hole,board,opponents,tightness,aggression)
    features=model_features(hole,board,opponents,tightness,aggression)
    return _model_from_features(features)

def _model_from_features(features, fast=False):
    opponents=features[1]
    # Preserve exact unbeatable royal flush / shared-board tie cases.
    if features[4] == 8 and features[5] == 14:
        return 1/(opponents+1) if features[32] else 1.
    if features[32] and features[31] == 7 and (features[38] == 14 or (features[37] == 14 and features[38] == 13)):
        return 1/(opponents+1)
    if features[32] and features[31] == 4 and features[37] == 14 and features[23] <= 2:
        return 1/(opponents+1)
    if _COMPILED_PREDICTOR is not None:
        predictor=_COMPILED_PREDICTOR_FAST if fast and _COMPILED_PREDICTOR_FAST is not None else _COMPILED_PREDICTOR
        return max(0.,min(1.,predictor(features)))
    total=_MODEL['base'];rate=_MODEL['rate']
    for tree in (_MODEL['trees'][:80] if fast else _MODEL['trees']):
        node=tree[0]
        while node[0]>=0:
            node=tree[node[2] if features[node[0]]<node[1] else node[3]]
        total+=rate*node[4]
    return max(0.,min(1.,total))

def preflop(hole, opponents=1, range_strength=0.0):
    opponents = max(1, min(7, int(opponents)))
    tightness = max(0., min(1., float(range_strength)))
    if _PREFLOP and hand_class(hole) in _PREFLOP['equities']:
        values = _PREFLOP['equities'][hand_class(hole)]
        axis = _PREFLOP['ranges']
        for i in range(1, len(axis)):
            if tightness <= axis[i]:
                w = (tightness - axis[i-1]) / (axis[i] - axis[i-1])
                return values[i-1][opponents-1]*(1-w) + values[i][opponents-1]*w
        return values[-1][opponents-1]
    hu = max(.12, min(.88, starting_score(hole) - .20 * tightness))
    return hu ** (1 + .71 * (opponents - 1))

def _heuristic(hole, board, opponents, tightness, aggression):
    """Bounded fallback model; intentionally not represented as Monte Carlo."""
    cards = list(hole) + list(board)
    rank = evaluate_python(cards)
    cat = rank >> 20
    hr = [c >> 2 for c in hole]
    br = [c >> 2 for c in board]
    highest = max(br)
    match = any(r in br for r in hr)
    pocket = hr[0] == hr[1]
    if len(board) == 5 and rank == evaluate_python(board):
        # Shared board value; do not mistake board quads/full houses for the nuts.
        high = (rank >> 16) & 15
        kicker=(rank >> 12)&15
        certain_tie = ((cat == 8 and high == 14) or (cat == 7 and (kicker == 14 or (high == 14 and kicker == 13)))
            or (cat == 4 and high == 14 and max(sum(c&3==s for c in board) for s in range(4))<=2))
        if certain_tie:
            return 1.0 / (opponents + 1)
        survival = {0:.02, 1:.08, 2:.22, 3:.16, 4:.68, 5:.40, 6:.72, 7:.68, 8:.95}[cat]
        return survival ** opponents / (opponents + 1)
    if cat == 0:
        base = .20 + .014 * max(hr) + (.055 if max(hr) == 12 else 0)
    elif cat == 1:
        if pocket and hr[0] > highest:
            base = .83
        elif match and highest in hr:
            kicker = max(r for r in hr if r != highest) if hr[0] != hr[1] else hr[0]
            base = .67 + .008 * kicker
        elif pocket:
            base = .45 + .019 * sum(r < hr[0] for r in br)
        elif match:
            base = .52 + .025 * sum(r < max(r for r in hr if r in br) for r in br)
        else:
            base = .24 + .012 * max(hr)
    elif cat == 2:
        base = .85 if pocket or (hr[0] in br and hr[1] in br) else .64 if match else .35
    elif cat == 3:
        base = .94 if pocket else .88 if match else .36
    else:
        base = {4:.91, 5:.96, 6:.985, 7:.997, 8:.999}[cat]
    if len(board) < 5 and cat < 4:
        suits = [sum(c & 3 == s for c in cards) for s in range(4)]
        flush_draw = any(suits[s] == 4 and any(c & 3 == s for c in hole) for s in range(4))
        mask = sum(1 << r for r in set(c >> 2 for c in cards))
        near_straight = any((mask & (31 << shift)).bit_count() == 4 for shift in range(9)) or (mask & 4111).bit_count() == 4
        draw = (.22 if len(board) == 3 else .12) if flush_draw else 0
        draw += (.12 if len(board) == 3 else .07) if near_straight else 0
        base += (1-base)*draw
    board_suits = [sum(c & 3 == s for c in board) for s in range(4)]
    if cat < 5 and max(board_suits) >= 3:
        base -= .07 * (max(board_suits)-2)
    base -= tightness * (.12 if cat < 2 else .05)
    base -= aggression * (.17 if cat < 2 else .08 if cat < 4 else .025)
    return max(.005, min(.999, base)) ** (1 + .80*(opponents-1))

def estimate(hole, board, opponents=1, range_strength=0.0, seed=17, samples=96, aggression=0.0, ranges=None, pots=None):
    """Estimate fractional showdown equity with hidden-card blockers.

    Native mode uses deterministic bounded Monte Carlo, with exact heads-up
    river enumeration. Without native code, postflop uses our offline-trained
    tree surrogate when available, then a lower-fidelity handcrafted heuristic.
    `ranges` can provide one `(preflop_tightness, aggression)` pair per opponent.
    All opponent holes are drawn before future community cards.
    `pots` contains optional `(amount, eligible_opponents_mask)` layers, each
    including hero. Return is expected fraction of their sum. Mask bits index
    the same opponent/ranges order. Do not include layers hero cannot win.
    """
    if len(hole) != 2 or len(board) not in (0,3,4,5):
        raise ValueError('invalid hole/board length')
    known = list(hole) + list(board)
    if len(set(known)) != len(known) or any(type(c) is not int or not 0 <= c < 52 for c in known):
        raise ValueError('duplicate or invalid card')
    opponents = max(1, min(7, int(opponents)))
    if ranges is not None:
        opponents = min(7, len(ranges))
        if opponents < 1:
            return 1.0
        tight = [max(0., min(1., float(x[0]))) for x in ranges[:opponents]]
        aggr = [max(0., min(1., float(x[1]))) for x in ranges[:opponents]]
    else:
        tight = [max(0., min(1., float(range_strength)))] * opponents
        aggr = [max(0., min(1., float(aggression)))] * opponents
    layers=None
    if pots is not None:
        grouped={}
        for amount,mask in pots:
            amount=float(amount)
            if not math.isfinite(amount) or amount<0 or type(mask) is not int or not 0<=mask<(1<<opponents):
                raise ValueError('invalid pot layer amount or eligibility mask')
            if amount:grouped[mask]=grouped.get(mask,0.)+amount
        if not grouped:return 0.
        layers=list(grouped.items())
        total=sum(grouped.values())
        if len(layers)>8 or not math.isfinite(total):raise ValueError('too many or invalid pot layers')
        if len(layers)==1 and layers[0][0]==(1<<opponents)-1:layers=None
        elif len(layers)==1 and layers[0][0]==0:return 1.
    if not board:
        # Homogeneous lookup approximation for unequal ranges, explicitly documented.
        if layers is not None:
            share=0.
            for mask,amount in layers:
                indices=[p for p in range(opponents) if mask&(1<<p)]
                share+=amount*(preflop(hole,len(indices),sum(tight[p] for p in indices)/len(indices)) if indices else 1.)
            return share/total
        return preflop(hole, opponents, sum(tight)/opponents)
    if _NATIVE is None:
        if layers is not None:
            share=0.
            features=model_features(hole,board,opponents,0.,0.) if _MODEL is not None else None
            for mask,amount in layers:
                indices=[p for p in range(opponents) if mask&(1<<p)]
                if not indices:share+=amount;continue
                n=len(indices);t=sum(tight[p] for p in indices)/n;a=sum(aggr[p] for p in indices)/n
                if features is not None:
                    # Four or more sidepot masks use the validated 80-tree prefix
                    # to bound pure-Python latency while preserving every layer.
                    features[1:4]=[n,t,a];value=_model_from_features(features,fast=len(layers)>3)
                else:value=_heuristic(hole,board,n,t,a)
                share+=amount*value
            return share/total
        return model_estimate(hole, board, opponents, sum(tight)/opponents, sum(aggr)/opponents)
    if len(board)==5 and opponents==1 and hasattr(_NATIVE,'poker_river_equity'):
        exact=_NATIVE.poker_river_equity((ctypes.c_int*2)(*hole),(ctypes.c_int*5)(*board),tight[0],aggr[0])
        if exact>=0:
            return exact if layers is None else sum(amount*(exact if mask else 1.) for mask,amount in layers)/total
    count = max(16, min(4096, int(samples)))
    if layers is not None and hasattr(_NATIVE,'poker_equity_pots'):
        return max(0.,min(1.,_NATIVE.poker_equity_pots((ctypes.c_int*2)(*hole),(ctypes.c_int*len(board))(*board),len(board),opponents,
            (ctypes.c_double*opponents)(*tight),(ctypes.c_double*opponents)(*aggr),int(seed)&0xffffffffffffffff,count,
            (ctypes.c_double*len(layers))(*(amount for mask,amount in layers)),(ctypes.c_int*len(layers))(*(mask for mask,amount in layers)),len(layers))))
    if layers is not None:
        # Backward-compatible old native library: evaluate each contested subset.
        return sum(amount*(estimate(hole,board,ranges=[(tight[p],aggr[p]) for p in range(opponents) if mask&(1<<p)],seed=seed,samples=count) if mask else 1.) for mask,amount in layers)/total
    result = _NATIVE.poker_equity((ctypes.c_int*2)(*hole), (ctypes.c_int*len(board))(*board), len(board), opponents,
        (ctypes.c_double*opponents)(*tight), (ctypes.c_double*opponents)(*aggr), int(seed)&0xffffffffffffffff, count)
    return max(0., min(1., result))

def native_equity(hole, board=(), opponents=1, range_strength=0., seed=17, samples=10000):
    """Offline generation helper; deliberately bypasses lookup and runtime cap."""
    if _NATIVE is None:
        raise RuntimeError('native accelerator required for offline generation')
    return _NATIVE.poker_equity((ctypes.c_int*2)(*hole), (ctypes.c_int*len(board))(*board), len(board), opponents,
        (ctypes.c_double*opponents)(*[range_strength]*opponents), (ctypes.c_double*opponents)(*[0.]*opponents), seed, samples)

def sampler_stats(reset=False):
    if _NATIVE is None or not hasattr(_NATIVE,'poker_stats'):
        return {}
    counts=(ctypes.c_uint64*6)()
    _NATIVE.poker_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_int]
    _NATIVE.poker_stats(counts,int(reset))
    result=dict(zip(('calls','trials','opponent_draws','proposal_cap_fallbacks','uniform_emergency_fallbacks','proposals'),map(int,counts)))
    if hasattr(_NATIVE,'poker_exact_calls'):
        _NATIVE.poker_exact_calls.argtypes=[ctypes.c_int];_NATIVE.poker_exact_calls.restype=ctypes.c_uint64
        result['exact_river_calls']=int(_NATIVE.poker_exact_calls(int(reset)))
    return result
