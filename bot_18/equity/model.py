"""Fast offline equity prediction. Models load once when this module is imported."""
from functools import lru_cache
import json
import hashlib
from pathlib import Path

from .features import features, FEATURE_VERSION
from .preflop import class_index, hand_score, range_weights
from .special import quads_on_board, straight_on_board


_DIRECTORY = Path(__file__).resolve().parent
try:
    _MODEL_BYTES = (_DIRECTORY / "model.json").read_bytes()
    _MODEL = json.loads(_MODEL_BYTES)
except FileNotFoundError:
    _MODEL = None
try:
    _PREFLOP = json.loads((_DIRECTORY / "preflop.json").read_text(encoding="utf-8"))["table"]
except FileNotFoundError:
    _PREFLOP = None
try:
    _PREMIUM = json.loads((_DIRECTORY / "premium.json").read_text(encoding="utf-8"))["table"]
except FileNotFoundError:
    _PREMIUM = None

MODEL_COMPATIBLE = _MODEL is not None and _MODEL.get("feature_version") == FEATURE_VERSION
MODEL_AVAILABLE = MODEL_COMPATIBLE
MODEL_STATUS = ("ready" if MODEL_COMPATIBLE else "missing" if _MODEL is None else "feature-version-mismatch")
PREFLOP_AVAILABLE = _PREFLOP is not None
PREMIUM_AVAILABLE = _PREMIUM is not None
_TREES = _MODEL.get("trees",()) if MODEL_COMPATIBLE else ()
_BASE = _MODEL.get("base",0.0) if MODEL_COMPATIBLE else 0.0
_STREETS = _MODEL.get("models",{}) if MODEL_COMPATIBLE else {}
_COMPILED = None
if MODEL_COMPATIBLE:
    try:
        from . import compiled as _compiled
        if (_compiled.FEATURE_VERSION == FEATURE_VERSION and
                _compiled.MODEL_SHA256 == hashlib.sha256(_MODEL_BYTES).hexdigest()):
            _COMPILED = _compiled.predict
    except ImportError:
        pass
COMPILED_AVAILABLE = _COMPILED is not None


@lru_cache(maxsize=128)
def _describe(hole, board):
    return tuple(features(hole, board))


def describe(hole, board):
    """Cached 75-feature tuple; made category is [7], board category is [13]."""
    return _describe(tuple(sorted(hole)), tuple(sorted(board)))


def forced_tie(hole, board):
    """Whether every possible remaining holding has exactly the same best hand."""
    if len(board) != 5:
        return False
    descriptor = describe(hole, board)
    return forced_tie_features(descriptor)


def forced_tie_features(descriptor):
    if descriptor[0] != 5:
        return False
    category, high, kicker = descriptor[13:16]
    return ((category == 8 and high == 14)
            or (category == 7 and kicker == (13 if high == 14 else 14))
            or (category == 4 and high == 14 and descriptor[30] <= 2))


def estimate_features(descriptor, range_strength=1.0, opponents=1):
    """Expected share of the pot versus the documented preflop range abstraction."""
    opponents = max(1, min(7, int(opponents)))
    range_strength = max(.06, min(1.0, float(range_strength)))
    # A royal flush on the board is an exact tie, including multiway.
    if forced_tie_features(descriptor):
        return 1.0 / (opponents + 1)
    if descriptor[7] == 8 and descriptor[8] == 14:
        return 1.0
    if not MODEL_AVAILABLE:
        # Runnable checkpoint only. This branch is explicitly reported as untrained.
        cat = int(descriptor[7])
        base = (.22, .62, .78, .88, .92, .96, .985, .997, .9999)[cat]
        if cat == 0:
            base += (descriptor[1] - 7) * .018
        if descriptor[33] == 4 and descriptor[0] < 5:
            base += .15 if descriptor[0] == 3 else .08
        base *= .82 + .18 * range_strength
        return max(.001, min(.999, base)) ** (1 + .65 * (opponents - 1))
    if opponents in (4, 6):
        # The training grid is 1,2,3,5,7. A histogram tree's midpoint split
        # treats 4 exactly like 3 (and 6 like 5); it does NOT interpolate.
        # Interpolate log pot-share between supported counts, preserving
        # positivity and the trained monotonic ordering. Exact ties above
        # and analytic rare-board overrides retain their exact formulas.
        lower = estimate_features(descriptor, range_strength, opponents - 1)
        upper = estimate_features(descriptor, range_strength, opponents + 1)
        return (lower * upper) ** .5
    row = (*descriptor, range_strength, opponents)
    if _COMPILED is not None:
        return max(.0001,min(.9999,_COMPILED(row)))
    part = _STREETS.get(str(descriptor[0])) if _STREETS else None
    result = part["base"] if part else _BASE
    trees = part["trees"] if part else _TREES
    for tree in trees:
        node = tree[0]
        while node[0] >= 0:
            node = tree[node[2] if row[node[0]] <= node[1] else node[3]]
        result += node[4]
    return max(.0001, min(.9999, result))


def preflop_equity(hole, range_strength=1.0, opponents=1):
    opponents = max(1, min(7, int(opponents)))
    if _PREFLOP is None:
        return max(.05, min(.86, hand_score(hole) - .13 * (1-range_strength))) ** (1 + .68 * (opponents-1))
    table = _PREFLOP[class_index(hole)]
    lo, hi, weight = range_weights(range_strength)
    return table[lo][opponents-1] * (1-weight) + table[hi][opponents-1] * weight


def estimate(hole, board, range_strength=1.0, opponents=1):
    if not board:
        return preflop_equity(hole, range_strength, opponents)
    descriptor=describe(hole,board)
    if len(board)==5 and descriptor[13]==7:
        exact=quads_on_board(hole,board,range_strength,opponents,descriptor)
        if exact is not None:return exact
    if len(board)==5 and descriptor[13]==4:
        exact=straight_on_board(hole,board,range_strength,opponents,descriptor)
        if exact is not None:return exact
    return estimate_features(descriptor, range_strength, opponents)


def estimate_premium(hole, nutted=False):
    """Heads-up equity versus QQ+/AK, or KK+ when nutted=True."""
    if _PREMIUM is None:
        return max(.04, min(.85, hand_score(hole) - (.28 if nutted else .22)))
    return _PREMIUM[class_index(hole)][int(bool(nutted))]
