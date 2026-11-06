"""Offline equity assets with deterministic, standard-library inference.

Model initialization occurs at import. A missing model uses a conservative
heuristic for development; packaged builds are checked to include both assets.
These estimates are conditioned on explicit synthetic ranges, not a solved game.
"""
import json
import hashlib
from pathlib import Path
from bot.features import VERSION, hand_index, make_features
from bot.evaluator import evaluate
from bot.exact import guaranteed_river_win

_ASSETS = Path(__file__).resolve().parent.parent / 'assets'
try:
    _MODEL = json.loads((_ASSETS / 'equity_model.json').read_text())
    if _MODEL['feature_version'] != VERSION:
        raise ValueError('Equity feature/model version mismatch')
    _TREES = _MODEL['trees']
    _BASE = _MODEL['baseline']
    MODEL_AVAILABLE = True
except FileNotFoundError:
    _TREES = []
    _BASE = 0.5
    MODEL_AVAILABLE = False
try:
    _PREFLOP = json.loads((_ASSETS / 'preflop.json').read_text())
    PREFLOP_AVAILABLE = True
except FileNotFoundError:
    _PREFLOP = None
    PREFLOP_AVAILABLE = False


try:
    _NARROW = json.loads((_ASSETS / 'preflop_narrow.json').read_text())
except FileNotFoundError:
    _NARROW = None


COMPILED_AVAILABLE = False
_COMPILED = None
if MODEL_AVAILABLE:
    try:
        from bot import _equity_compiled
        if (_equity_compiled.FEATURE_VERSION == VERSION and
                _equity_compiled.MODEL_SHA256 == hashlib.sha256((_ASSETS / 'equity_model.json').read_bytes()).hexdigest()):
            _COMPILED = _equity_compiled.predict
            COMPILED_AVAILABLE = True
    except ImportError:
        pass


def _fallback_strength(hole):
    a, b = sorted((hole[0] // 4 + 2, hole[1] // 4 + 2), reverse=True)
    if a == b:
        return 0.50 + a * 0.025
    return min(0.70, 0.20 + a * 0.022 + b * 0.011 + (0.035 if hole[0] % 4 == hole[1] % 4 else 0))


def percentile(hole):
    """Combo-weighted heads-up equity rank, smaller means stronger."""
    if _PREFLOP is not None:
        return _PREFLOP['percentile'][hand_index(hole)]
    return max(0.002, min(0.998, (0.75 - _fallback_strength(hole)) * 2.7))


def _range_bracket(fraction):
    fractions = (0.1, 0.25, 0.5, 1.0)
    if fraction <= 0.1:
        return 0, 0, 0.0
    for j in range(1, 4):
        if fraction <= fractions[j]:
            return j - 1, j, (fraction - fractions[j - 1]) / (fractions[j] - fractions[j - 1])
    return 3, 3, 0.0


def preflop(hole, range_fraction=1.0, opponents=1):
    n = max(1, min(7, int(opponents)))
    if _PREFLOP is None:
        return max(0.02, _fallback_strength(hole) - 0.14 * (1 - range_fraction)) ** (n ** 0.80)
    index = hand_index(hole)
    row = _PREFLOP['equity'][index]
    if _NARROW is not None and range_fraction < 0.1:
        nr = _NARROW['equity'][index]
        def value(k):
            if n <= 3:
                return nr[k * 3 + n - 1]
            e2, e3 = nr[k * 3 + 1], nr[k * 3 + 2]
            return e3 * min(1.0, e3 / max(0.00001, e2)) ** (n - 3)
        if range_fraction <= .025:
            return value(0)
        if range_fraction < .05:
            a, b = value(0), value(1)
            return a + (range_fraction - .025) / .025 * (b - a)
        a, b = value(1), row[n - 1]
        return a + (range_fraction - .05) / .05 * (b - a)
    lo, hi, t = _range_bracket(range_fraction)
    a, b = row[lo * 7 + n - 1], row[hi * 7 + n - 1]
    return a + t * (b - a)


def predict_reference(features):
    total = _BASE
    for tree in _TREES:
        node = tree[0]
        while node[0] >= 0:
            node = tree[node[2] if features[node[0]] <= node[1] else node[3]]
        total += node[1]
    return max(0.0, min(1.0, total))


def predict(features):
    if _COMPILED is not None:
        return max(0.0, min(1.0, _COMPILED(features)))
    return predict_reference(features)


def postflop(hole, board, range_fraction=1.0, opponents=1, pressure=0.0, features=None):
    n = max(1, min(7, int(opponents)))
    # These two royal-flush cases are exact regardless of opponent range.
    score = evaluate(list(hole) + list(board))
    if score == (8 << 20) | (14 << 16):
        if len(board) == 5 and evaluate(board) == score:
            return 1.0 / (n + 1)
        return 1.0
    if features is None:
        features = make_features(hole, board, n, max(0.1, min(1.0, range_fraction)), max(0.0, min(2.0, pressure)))
    else:
        features = list(features)
        features[-3:] = [n, max(0.1, min(1.0, range_fraction)), max(0.0, min(2.0, pressure))]
    if VERSION == 3 and guaranteed_river_win(features):
        return 1.0
    if MODEL_AVAILABLE:
        result = predict(features)
        # If hero plays the board, every opponent can at least tie hero.
        if len(board) == 5 and score == evaluate(board):
            result = min(result, 1.0 / (n + 1))
        return result
    category = score >> 20
    e = (0.25, 0.55, 0.73, 0.84, 0.90, 0.94, 0.98, 0.995, 0.999)[category]
    return max(0.02, e - 0.10 * pressure) ** (n ** 0.82)
