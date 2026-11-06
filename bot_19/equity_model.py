"""Offline equity assets with a small standard-library inference runtime.

Missing assets use explicit deterministic heuristic fallbacks, reported by
``asset_status()``. The model is a range-equity approximation, not a solver.
"""

import json
import math
import hashlib
import importlib.util
from struct import pack as _pack, unpack as _unpack
from pathlib import Path

from poker_eval import FEATURE_NAMES, features
from equity_features import MODEL_FEATURE_NAMES, board_plays, extend_features

_ROOT = Path(__file__).resolve().parent
_MODEL = None
_PREFLOP = None
_LOAD_ERRORS = []
_FAST = None
_FAST_ERROR = None
_MODEL_SHA256 = None


def _validate_model(data):
    if data.get("feature_names") not in (list(FEATURE_NAMES), list(MODEL_FEATURE_NAMES)):
        raise ValueError("model feature schema mismatch")
    streets = data["streets"]
    if set(streets) != {"3", "4", "5"}:
        raise ValueError("incomplete street models")
    derived = len(data["feature_names"]) > len(FEATURE_NAMES)
    if derived and (data.get("input_feature_names") != list(FEATURE_NAMES) or data.get("derived_version") != 1):
        raise ValueError("derived feature schema mismatch")
    for model in streets.values():
        if bool(model.get("derived")) != derived:
            raise ValueError("inconsistent derived feature declaration")
        if not math.isfinite(model["base"]) or not model["trees"]:
            raise ValueError("invalid base or empty forest")
        immutable = []
        for tree in model["trees"]:
            if not tree:
                raise ValueError("empty tree")
            for node_id, entry in enumerate(tree):
                if not isinstance(entry[0], int) or not math.isfinite(entry[1]):
                    raise ValueError("nonfinite model value")
                if entry[0] < 0:
                    if len(entry) != 2:
                        raise ValueError("malformed leaf")
                elif not (len(entry) == 4 and 0 <= entry[0] < len(data["feature_names"]) and
                          node_id < entry[2] < len(tree) and node_id < entry[3] < len(tree)):
                    raise ValueError("malformed or cyclic tree")
                else:
                    entry[1] = _unpack("f", _pack("f", entry[1]))[0]
            immutable.append(tuple(tuple(entry) for entry in tree))
        model["trees"] = tuple(immutable)
    return data


def _validate_preflop(data):
    if len(data["base"]) != 169 or len(data["percentile"]) != 169:
        raise ValueError("preflop class count")
    widths = data["widths"]
    if len(widths) < 2 or widths != sorted(set(widths)) or widths[0] > .05 or widths[-1] != 1:
        raise ValueError("preflop width grid")
    if len(data["equity"]) != 169 * len(widths) * 7:
        raise ValueError("preflop cell count")
    for key in ("base", "percentile", "equity"):
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in data[key]):
            raise ValueError("invalid preflop probability")
        data[key] = tuple(data[key])
    data["widths"] = tuple(widths)
    return data


for _name, _file in (("model", "equity_model.json"), ("preflop", "preflop.json")):
    try:
        _raw = (_ROOT / "assets" / _file).read_bytes()
        _data = json.loads(_raw)
        if _data.get("schema") != 1:
            raise ValueError("unsupported asset schema")
        if _name == "model":
            _MODEL = _validate_model(_data)
            _MODEL_SHA256 = hashlib.sha256(_raw).hexdigest()
        else:
            _PREFLOP = _validate_preflop(_data)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as _error:
        _LOAD_ERRORS.append(f"{_file}: {type(_error).__name__}")

if _MODEL is not None:
    try:
        _spec = importlib.util.spec_from_file_location("_bot19_equity_fast", _ROOT / "assets" / "equity_fast.py")
        _fast_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_fast_module)
        if _fast_module.SOURCE_SHA256 != _MODEL_SHA256 or _fast_module.SCHEMA != 1:
            raise ValueError("generated/source model mismatch")
        _FAST = _fast_module.MODELS
    except (OSError, ValueError, AttributeError, ImportError, SyntaxError) as _error:
        _FAST_ERROR = type(_error).__name__


def asset_status():
    return {"postflop_model": _MODEL is not None, "preflop_table": _PREFLOP is not None,
            "load_errors": list(_LOAD_ERRORS),
            "inference_backend": "generated" if _FAST is not None else "trees" if _MODEL is not None else "heuristic",
            "fast_error": _FAST_ERROR}


def preflop_class(hole):
    a, b = hole
    hi, lo = max(a >> 2, b >> 2), min(a >> 2, b >> 2)
    return hi * 13 + lo if a & 3 == b & 3 else lo * 13 + hi


def _fallback_base(hole):
    a, b = hole
    hi, lo = max(a >> 2, b >> 2) + 2, min(a >> 2, b >> 2) + 2
    if hi == lo:
        return .50 + .029 * (hi - 2)
    return min(.69, .29 + .019 * hi + .007 * lo + .025 * (a & 3 == b & 3) - .007 * max(0, hi - lo - 2))


def preflop_percentile(hole):
    """Top fraction of 1,326 holdings; AA is near zero, weakest near one."""
    if _PREFLOP is not None:
        return _PREFLOP["percentile"][preflop_class(hole)]
    return max(.002, min(.999, 1.0 - (_fallback_base(hole) - .32) / .50))


def preflop_equity(hole, width=1.0, opponents=1):
    """Interpolate table showdown share against 1..7 blocked range draws."""
    n = max(1, min(7, int(opponents)))
    w = max(.05, min(1.0, float(width)))
    if _PREFLOP is None:
        base = _fallback_base(hole)
        tight_penalty = .12 * (1 - w) * (1.0 - base)
        return max(.005, min(.995, (base - tight_penalty) ** (n ** .78)))
    widths = _PREFLOP["widths"]
    k = 0
    while k < len(widths) - 2 and w > widths[k + 1]:
        k += 1
    alpha = (w - widths[k]) / (widths[k + 1] - widths[k])
    offset = (preflop_class(hole) * len(widths) + k) * 7 + n - 1
    values = _PREFLOP["equity"]
    return values[offset] * (1 - alpha) + values[offset + 7] * alpha


def _fallback_equity(f):
    cat, top, boardcat = int(f[6]), f[7], int(f[34])
    if cat >= 6:
        q = .985 if cat > boardcat else .52
    elif cat == 5:
        q = .965 if f[23] else .52
    elif cat == 4:
        q = .90 if f[38] else .50
    elif cat == 3:
        q = .86 if cat > boardcat else .45 + .015 * (f[1] - 8)
    elif cat == 2:
        q = .79 if boardcat == 0 else .66 if boardcat == 1 else .42
    elif cat == 1:
        if f[3]:
            q = .76 if f[31] == 0 else .59 if f[31] == 1 else .43
        elif f[29] or f[30]:
            q = .70 if top >= f[12] else .54 if top >= f[13] else .42
        else:
            q = .30 + .015 * (f[1] - 8)
    else:
        q = .19 + .018 * (f[1] - 8) + .035 * f[28]
    if f[0] < 5:
        if f[22] == 4 and f[23] > 0:
            q += .13 if f[0] == 3 else .08
        if f[39] and f[27]:
            q += .045 * min(2, f[27])
    q -= (.06 * (1 - f[41]) + .16 * f[42]) * (1.02 - q)
    return max(.01, min(.99, q))


def _predict_reference(f):
    """Reference interpreter retained for offline equivalence validation."""
    if _MODEL is None:
        return _fallback_equity(f)
    f = list(f)
    f[41], f[42] = _unpack("ff", _pack("ff", f[41], f[42]))
    model = _MODEL["streets"][str(int(f[0]))]
    if model.get("derived"):
        f = extend_features(f)
    value = model["base"]
    for tree in model["trees"]:
        node = 0
        while True:
            entry = tree[node]
            index = entry[0]
            if index < 0:
                value += entry[1]
                break
            node = entry[2] if f[index] < entry[1] else entry[3]
    return max(.001, min(.999, value))


def predict_from_features(f):
    """Predict heads-up pot share; last two features are width/aggression."""
    if _FAST is not None:
        return _FAST[int(f[0])](f)
    return _predict_reference(f)


def predict_equity(hole, board, range_width=1.0, aggression=0.0):
    return predict_from_features(features(hole, board, range_width, aggression))


# Selected on 2,400 synthetic cases (seed1907719), then validated on 1,800
# untouched cases (seed1907720,8192 MC draws). See the calibration report.
_MULTI_CONCENTRATION = (
    ((1.5, .75), (4., 1.5), (12., 4.)),
    ((3., .75), (8., 3.), (20., 8.)),
)


def multiway_equity(hu, f, opponents):
    """Cheap calibrated identical-range approximation, including board draws.

    Flop/turn use a beta moment for correlated runout outcomes. Fractional
    opponent counts interpolate adjacent moments. River uses fitted powers.
    Different opponent ranges and side pots still require policy handling.
    """
    q = max(0., min(1., hu))
    n = max(1., min(7., opponents))
    if board_plays(f):
        # Hero cannot win outright. q=.5*P(tie) heads-up, and must tie every
        # opponent to receive 1/(n+1). The exponent still approximates joint
        # card removal/independence; only guaranteed-tie q=.5 is exact.
        q = min(q, .5)
        return (2*q) ** (n ** 1.05) / (n+1)
    if n == 1:
        return q
    cat = int(f[6])
    if cat == 1 and not (f[3] or f[29] or f[30]):
        cat = 0
    cat = min(cat, 2)
    street = int(f[0])
    if street == 5:
        return q ** (n ** (1.025 if cat == 2 else 1.05))
    draw = bool((f[22] == 4 and f[23]) or (f[27] >= 2 and f[39]))
    concentration = _MULTI_CONCENTRATION[street-3][cat][draw]
    result = q
    whole = int(n)
    for k in range(1, whole):
        result *= (q*concentration+k)/(concentration+k)
    fractional = n-whole
    if fractional:
        result *= 1-fractional+fractional*(q*concentration+whole)/(concentration+whole)
    return result
