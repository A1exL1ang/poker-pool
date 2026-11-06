"""Offline equity estimates with a dependency-free, bounded runtime.

Postflop estimates come from compact boosted trees trained on our own native
Monte Carlo simulations. Preflop estimates interpolate an independent 169-hand
table. Ranges are a model of play, not a solved strategy or opponent knowledge.
Missing assets retain a conservative heuristic fallback for adapter bring-up.
"""

from __future__ import annotations

import bisect
import gzip
import json
import math
from functools import lru_cache
from pathlib import Path

from .features import extract_features, FEATURE_NAMES, FEATURE_VERSION

ASSET_DIRECTORY = Path(__file__).with_name("assets")


def hand_class(hole) -> int:
    """169-cell index: low/high offsuit, high/low suited, diagonal pairs."""
    a, b = hole
    high = max(a >> 2, b >> 2)
    low = min(a >> 2, b >> 2)
    return high * 13 + low if a & 3 == b & 3 else low * 13 + high


def _fallback_order():
    scores = []
    for index in range(169):
        row, col = divmod(index, 13)
        high, low = max(row, col) + 2, min(row, col) + 2
        if row == col:
            score = .48 + .027 * high
        else:
            score = .2 + .024 * high + .011 * low + .035 * (row > col) - .007 * max(0, high - low - 1)
        scores.append(score)
    percentiles = [0.0] * 169
    cumulative = 0
    for i in sorted(range(169), key=lambda i: scores[i], reverse=True):
        row, col = divmod(i, 13)
        count = 6 if row == col else (4 if row > col else 12)
        percentiles[i] = (cumulative + .5 * count) / 1326
        cumulative += count
    return {"percentiles": percentiles, "strengths": scores}


def _load_preflop(path=ASSET_DIRECTORY / "preflop.json"):
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if data.get("version") != 1 or len(data["percentiles"]) != 169 or len(data["equities"]) != 169 * len(data["fractions"]) * 7:
            raise ValueError("Invalid preflop table schema")
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return _fallback_order()


_PREFLOP = _load_preflop()


def _compile_trees(base, trees, feature_count):
    """Compile numeric, validated tree data once, before timed decisions.

    No source text from an observation or model field is executed. Only bounded
    integer feature indices, finite float constants, and fixed syntax enter
    the generated function. This avoids a Python loop/tuple unpack per node.
    """
    if not math.isfinite(base) or sum(map(len, trees)) > 200000:
        raise ValueError("Invalid or oversized equity model")
    expressions = []
    for tree in trees:
        if not tree:
            raise ValueError("Empty equity tree")
        for node in tree:
            if len(node) != 5 or type(node[0]) is not int:
                raise ValueError("Malformed equity tree node")
            feature, threshold, left, right, value = node
            if feature < -1 or feature >= feature_count or not math.isfinite(float(threshold)) or not math.isfinite(float(value)):
                raise ValueError("Invalid equity tree scalar")
            if feature >= 0 and (type(left) is not int or type(right) is not int or not 0 <= left < len(tree) or not 0 <= right < len(tree)):
                raise ValueError("Invalid equity tree child")

        def expression(index, ancestors):
            if index in ancestors or len(ancestors) > 64:
                raise ValueError("Cyclic or too-deep equity tree")
            feature, threshold, left, right, value = tree[index]
            if feature == -1:
                return repr(float(value))
            path = ancestors | {index}
            return "(" + expression(left, path) + " if x[" + str(feature) + "] <= " + repr(float(threshold)) + " else " + expression(right, path) + ")"

        expressions.append(expression(0, set()))
    source = "def predict_vector(x):\n    return (" + repr(float(base)) + "\n" + "".join("        + " + item + "\n" for item in expressions) + "    )\n"
    namespace = {"__builtins__": {}}
    exec(compile(source, "<validated-equity-trees>", "exec"), namespace)
    return namespace["predict_vector"]


@lru_cache(maxsize=4)
def _read_model(path: str, stamp: tuple):
    """Share immutable tree tuples across independent table-local agents.

    The size/mtime cache key also permits deliberate model replacement during
    offline experiments without retaining a stale tree set in the process.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as file:
        data = json.load(file)
    feature_count = len(data["feature_names"])
    compatible = data["feature_version"] == FEATURE_VERSION and data["feature_names"] == list(FEATURE_NAMES)
    compatible |= data["feature_version"] == 1 and feature_count == 64 and data["feature_names"] == list(FEATURE_NAMES[:64])
    if data.get("format_version") not in (1, 2) or not compatible:
        raise ValueError("Equity feature/model schema mismatch")
    if data["query_features"] != ["opponents", "range_fraction", "pressure", "preflop_percentile"]:
        raise ValueError("Equity query/model schema mismatch")
    def unpack(model):
        trees = tuple(tuple(tuple(node) for node in tree) for tree in model["trees"])
        base = float(model["base"])
        return base, trees, _compile_trees(base, trees, feature_count + 4)
    if data["format_version"] == 1:
        model = unpack(data)
        return (None, None, None, model, model, model), feature_count
    return (None, None, None, *(unpack(data["models"][str(n)]) for n in (3, 4, 5))), feature_count


def preflop_percentile(hole) -> float:
    """Fraction of combinations above this class; lower is stronger."""
    return _PREFLOP["percentiles"][hand_class(hole)]


def _preflop_value(table, index: int, opponents: int, fraction: float) -> float:
    n = min(7, max(1, int(opponents)))
    if "equities" not in table:
        headsup = table["strengths"][index]
        headsup = max(.08, headsup - .17 * (1 - fraction))
        return headsup ** (n ** .78)
    fractions = table["fractions"]
    j = bisect.bisect_left(fractions, fraction)
    if j <= 0:
        return table["equities"][(index * len(fractions)) * 7 + n - 1]
    if j >= len(fractions):
        j = len(fractions) - 1
        return table["equities"][(index * len(fractions) + j) * 7 + n - 1]
    # Log-fraction interpolation follows the density of the training grid.
    weight = math.log(fraction / fractions[j - 1]) / math.log(fractions[j] / fractions[j - 1])
    lo = table["equities"][(index * len(fractions) + j - 1) * 7 + n - 1]
    hi = table["equities"][(index * len(fractions) + j) * 7 + n - 1]
    return lo + weight * (hi - lo)


def preflop_equity(hole, opponents=1, range_fraction=1.0) -> float:
    if opponents <= 0:
        return 1.0
    return _preflop_value(_PREFLOP, hand_class(hole), opponents, min(1.0, max(.05, range_fraction)))


class EquityEstimator:
    """Initialize before timed play, then reuse extracted features per action."""

    def __init__(self, model_path=None, preflop_path=None):
        self.preflop = _PREFLOP if preflop_path is None else _load_preflop(preflop_path)
        self.base = 0.0
        self.trees = ()
        self.model_loaded = False
        self._models = ()
        self.feature_count = len(FEATURE_NAMES)
        self.preflop_loaded = "equities" in self.preflop
        path = ASSET_DIRECTORY / "equity_model.json.gz" if model_path is None else Path(model_path)
        try:
            info = path.stat()
            self._models, self.feature_count = _read_model(str(path.resolve()), (info.st_size, info.st_mtime_ns))
            self.base, self.trees, _ = self._models[3]
            self.model_loaded = True
        except (OSError, ValueError, KeyError, TypeError):
            self.base, self.trees = 0.0, ()

    def estimate(self, hole, board, opponents=1, range_fraction=1.0, pressure=0.0) -> float:
        if opponents <= 0:
            return 1.0
        if not board:
            return _preflop_value(self.preflop, hand_class(hole), opponents, min(1.0, max(.05, range_fraction)))
        return self.predict(extract_features(hole, board), opponents, range_fraction, pressure)

    def predict(self, features, opponents=1, range_fraction=1.0, pressure=0.0) -> float:
        if opponents <= 0:
            return 1.0
        opponents = min(7, max(1, int(opponents)))
        fraction = min(1.0, max(.05, float(range_fraction)))
        pressure = min(1.0, max(0.0, float(pressure)))
        # A royal flush can be resolved without statistical approximation.
        if features[6] == 8 and features[7] == 14:
            return 1.0 / (opponents + 1) if features[45] else 1.0
        if not self.trees:
            return self._fallback(features, opponents, fraction, pressure)
        high, low = int(features[1]) - 2, int(features[2]) - 2
        index = high * 13 + low if features[4] else low * 13 + high
        vector = (*features[:self.feature_count], opponents, fraction, pressure, self.preflop["percentiles"][index])
        prediction = self._models[int(features[0])][2](vector)
        return min(1.0, max(0.0, prediction))

    @staticmethod
    def _fallback(f, n, fraction, pressure):
        category, board_category = int(f[6]), int(f[12])
        equity = (.24, .61, .81, .9, .94, .96, .985, .995, .999)[category]
        if category <= board_category:
            equity = .23 + .022 * max(0, f[1] - 8)
        elif category == 1:
            equity -= .09 * min(3, f[34])
        if f[0] < 5:
            draw = (.14 if f[35] == 4 and f[36] else 0) + .045 * min(2, f[41])
            equity = min(.94, equity + draw)
        equity = min(.999, max(.01, equity - .12 * (1 - fraction) - .19 * pressure))
        return equity ** (n ** .77)


_DEFAULT = None


def estimate_equity(hole, board, opponents=1, range_fraction=1.0, pressure=0.0):
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = EquityEstimator()
    return _DEFAULT.estimate(hole, board, opponents, range_fraction, pressure)
