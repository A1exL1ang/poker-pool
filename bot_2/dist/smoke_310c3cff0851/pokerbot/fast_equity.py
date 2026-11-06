"""Pure-Python inference for the locally trained synthetic equity surrogate.

The model estimates range-conditioned showdown share, not an equilibrium
strategy. Every model asset is reproducible from tools/train_equity.cpp.
"""
import json
import hashlib
from pathlib import Path
from .cards import (BITS, COUNT, STRAIGHT, HIGH, class_index, board_state,
                    evaluate_two, evaluate)
from .equity import PERCENTILE, PREFLOP, CUTOFFS, RANGES, range_index, _likelihood
from ._equity_compiled import predict_raw, SOURCE_SHA256

_source = (Path(__file__).resolve().parent.parent / "assets" / "equity_model.json").read_bytes()
if hashlib.sha256(_source).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("equity model/source mismatch: run python tools/compile_equity.py")
_model = json.loads(_source)
BASE = tuple(_model["base"])
RATE = _model["rate"]
DEPTH = _model["depth"]
METADATA = {k: v for k, v in _model.items() if k != "trees"}
del _model, _source


def features(hole, board, ri, pressure, state=None, score=None):
    a, b = hole
    if state is None:
        state = board_state(board)
    if score is None:
        score = evaluate_two(state, a, b)
    mask, two, three = state[:3]
    rlo, rhi = sorted((a // 4 + 2, b // 4 + 2))
    rs = sorted((c // 4 + 2 for c in board), reverse=True)
    rs += [0] * (5 - len(rs))
    h = class_index(a, b)
    allmask = mask | BITS[a] | BITS[b]
    outs = 0
    if not STRAIGHT[allmask]:
        outs = sum(not allmask & (1 << r) and bool(STRAIGHT[allmask | (1 << r)]) for r in range(13))
    window = max(max(COUNT[(allmask >> k) & 31] for k in range(9)), COUNT[allmask & 0x100F])
    bwindow = max(max(COUNT[(mask >> k) & 31] for k in range(9)), COUNT[mask & 0x100F])
    suits = list(state[4:])
    suits[a & 3] |= BITS[a]
    suits[b & 3] |= BITS[b]
    flen = max(COUNT[s] for s in suits)
    fhigh = max((c // 4 + 2 for c in hole if COUNT[suits[c & 3]] == flen), default=0)
    hi, lo = (a, b) if a // 4 >= b // 4 else (b, a)
    return (len(board) - 2, rlo, rhi, int(a // 4 == b // 4), int(a % 4 == b % 4),
            int(PERCENTILE[h] * 255), int(PREFLOP[h * 10 + ri] * 255),
            int(CUTOFFS[ri] * 255), int(pressure * 255), *rs,
            max(COUNT[s] for s in state[4:]), COUNT[two], COUNT[three], score >> 20,
            *((score >> (16 - 4 * k)) & 15 for k in range(5)),
            flen, fhigh, outs, window,
            rhi if mask & (1 << (rhi - 2)) else 0,
            rlo if mask & (1 << (rlo - 2)) else 0,
            int(a // 4 + 2 > rs[0]) + int(b // 4 + 2 > rs[0]),
            bwindow, COUNT[mask], COUNT[state[4 + (hi & 3)]], COUNT[state[4 + (lo & 3)]])


def predict(x, opponents):
    # Models contain joint-runout shares for 1,2,3,5,7 opponents.
    n = max(1, min(7, opponents))
    hu, eq = predict_raw(x, n - 1)
    hu = max(0.0, min(1.0, hu))
    return max(0.0, min(hu, eq)), hu


def equity(hole, board, width, opponents, pressure, state=None):
    if state is None:
        state = board_state(board)
    made = evaluate_two(state, *hole)
    # Exact boundary conditions guard very rare hands and board-only chops.
    if made == 0x8E0000:
        if len(board) == 5 and evaluate(board) == made:
            return 1 / (opponents + 1), .5, made
        return 1.0, 1.0, made
    ri = range_index(width)
    # Narrow river ranges are cheaper to enumerate than to approximate. This
    # is exact HU equity over the modeled range, with exact split accounting.
    if len(board) == 5 and len(RANGES[ri]) <= 100:
        known = sum(1 << c for c in hole + tuple(board))
        win = tie = total = 0.0
        for a, b in RANGES[ri]:
            if known & ((1 << a) | (1 << b)):
                continue
            score = evaluate_two(state, a, b)
            w = _likelihood(score, state, a, b, pressure, 3)
            total += w
            if made > score:
                win += w
            elif made == score:
                tie += w
        if total:
            pw, pt = win / total, tie / total
            eq = pw ** opponents if pt < 1e-12 else ((pw + pt) ** (opponents + 1) - pw ** (opponents + 1)) / ((opponents + 1) * pt)
            return eq, pw + .5 * pt, made
    eq, hu = predict(features(hole, board, ri, pressure, state, made), opponents)
    return eq, hu, made
