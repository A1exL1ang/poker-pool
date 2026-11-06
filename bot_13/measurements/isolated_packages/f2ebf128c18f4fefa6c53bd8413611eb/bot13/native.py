"""ctypes access to the optional original C++ evaluator and equity sampler.

No compilation, filesystem writes, network calls, or subprocesses occur on import
or during gameplay. Build ahead of packaging with ``python native/build.py``.
"""
from __future__ import annotations
import ctypes as C
import math
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_I = C.c_int
_D = C.c_double
_U = C.c_uint32
_I2 = _I * 2
_D2 = _D * 2
_LIB = None
_LOAD_ERROR = None
try:
    _LIB = C.CDLL(str(_ROOT / "native" / ("bot13.dll" if sys.platform == "win32" else "libbot13.so")))
    _LIB.b13_evaluate.argtypes = [C.POINTER(_I), _I]
    _LIB.b13_evaluate.restype = _U
    _LIB.b13_evaluate_many.argtypes = [C.POINTER(_I), _I, _I, C.POINTER(_U)]
    _LIB.b13_evaluate_many.restype = _I
    _LIB.b13_equity.argtypes = [C.POINTER(_I), C.POINTER(_I), _I, _I, C.POINTER(_D), C.c_uint64, _I, C.POINTER(_D)]
    _LIB.b13_equity.restype = _I
    _LIB.b13_model_equity.argtypes = [C.POINTER(_I), C.POINTER(_I), _I, _I, C.POINTER(_D), C.c_uint64, _I, C.POINTER(_D)]
    _LIB.b13_model_equity.restype = _I
    _LIB.b13_set_preflop_strength.argtypes = [C.POINTER(_D)]
    _LIB.b13_set_preflop_strength.restype = _I
    _LIB.b13_icm_equity.argtypes = [C.POINTER(_D), C.POINTER(_D), _I, _I]
    _LIB.b13_icm_equity.restype = _D
    _LIB.b13_five_card_histogram.argtypes = [C.POINTER(C.c_uint64)]
    _LIB.b13_five_card_histogram.restype = _I
except (OSError, AttributeError) as exc:
    _LOAD_ERROR = str(exc)
    _LIB = None

native_available = _LIB is not None
load_error = _LOAD_ERROR


class NativeEvaluator:
    """Explicit native-only backend. Unsupported/invalid inputs raise ValueError.

    ``weights`` is a contiguous ctypes double array (opponents*1326 entries),
    or a Python flat iterable converted for this call. Reuse the ctypes array
    when making decisions. No weights means independent uniformly random hands.
    """
    available = native_available

    def __init__(self):
        if _LIB is None:
            raise RuntimeError("Optional bot13 native backend unavailable: " + str(_LOAD_ERROR))
        self.last_accepted = 0

    def evaluate(self, cards):
        values = tuple(cards)
        if not 5 <= len(values) <= 7:
            raise ValueError("evaluate requires 5..7 distinct cards in 0..51")
        score = int(_LIB.b13_evaluate((_I * len(values))(*values), len(values)))
        if not score:
            raise ValueError("evaluate requires 5..7 distinct cards in 0..51")
        return score

    def evaluate_many(self, hands):
        hands = list(hands)
        if not hands:
            return []
        n = len(hands[0])
        if not 5 <= n <= 7 or any(len(h) != n for h in hands):
            raise ValueError("all hands must have the same length 5..7")
        flat = (_I * (len(hands)*n))(*(c for h in hands for c in h))
        scores = (_U * len(hands))()
        if _LIB.b13_evaluate_many(flat, n, len(hands), scores) != len(hands) or any(s == 0 for s in scores):
            raise ValueError("invalid cards")
        return list(scores)

    def equity(self, hero, board=(), opponents=1, weights=None, seed=1, samples=64):
        if len(hero) != 2 or len(board) not in (0, 3, 4, 5):
            raise ValueError("hero requires two cards; board length must be 0,3,4,5")
        if not 1 <= opponents <= 7 or not 0 <= samples <= 10000000 or (samples == 0 and not (len(board) == 5 and opponents == 1)):
            raise ValueError("opponents 1..7; samples 1..10000000 (0 allowed for exact HU river)")
        if weights is not None:
            if len(weights) != opponents * 1326:
                raise ValueError("weights requires opponents*1326 entries")
            if not isinstance(weights, C.Array) or getattr(weights, "_type_", None) is not _D:
                weights = (_D * len(weights))(*weights)
        out = _D2()
        accepted = _LIB.b13_equity(_I2(*hero), (_I * len(board))(*board), len(board), opponents, weights, seed & ((1 << 64)-1), samples, out)
        self.last_accepted = accepted
        if accepted <= 0:
            raise ValueError("native equity rejected input or has insufficient disjoint range support (status %d)" % accepted)
        return out[0], out[1]

    def modeled_equity(self, hero, board, specs, seed=1, samples=64):
        """Equity vs inferred ranges, with shared-rollout sidepot prefix shares.

        Each opponent spec has five doubles: (preflop_top_fraction,
        postflop_strength_threshold, postflop_slope, bluff_floor, draw_weight).
        A nonpositive postflop threshold disables postflop filtering. Opponents
        remain in caller order; ``prefixes[k-1]`` is equity vs first k opponents.
        All calculations use fixed bounded samples, except exact HU river.
        """
        if len(hero) != 2 or len(board) not in (0, 3, 4, 5) or not 1 <= len(specs) <= 7:
            raise ValueError("two hero cards; board 0,3,4,5; 1..7 opponents")
        if any(len(spec) != 5 for spec in specs):
            raise ValueError("every opponent model requires five parameters")
        opponents = len(specs)
        if not 0 <= samples <= 10000000 or (samples == 0 and not (len(board) == 5 and opponents == 1)):
            raise ValueError("invalid sample count")
        parameters = (_D * (opponents*5))(*(x for spec in specs for x in spec))
        out = (_D * (6+2*opponents))()
        accepted = _LIB.b13_model_equity(_I2(*hero), (_I * len(board))(*board), len(board), opponents, parameters, seed & ((1 << 64)-1), samples, out)
        self.last_accepted = accepted
        if accepted <= 0:
            raise ValueError("modeled equity rejected input or has insufficient range support (status %d)" % accepted)
        return {"equity": out[0], "stderr": out[1], "strength": out[2],
                "flush_draw": out[3], "straight_draw": out[4],
                "accepted": accepted, "accepted_fraction": out[5],
                "prefixes": tuple(out[6:6+opponents]),
                "prefix_stderr": tuple(out[6+opponents:6+2*opponents])}

    def set_preflop_strength(self, scores):
        """Set learned native combo ordering once, from 169 canonical hand keys.

        ``scores`` maps e.g. AA, AKo, AKs to HU equity, or is the generated table's
        ``hands`` mapping (entries have ``random`` arrays). This is initialization
        work, outside decision time. The module state is shared across instances.
        """
        rank = "23456789TJQKA"
        values = []
        for hi in range(12, -1, -1):
            for lo in range(hi, -1, -1):
                for suited in range(1 if hi == lo else 2):
                    key = rank[hi] + rank[lo] + ("" if hi == lo else ("s" if suited else "o"))
                    value = scores[key]
                    if isinstance(value, dict):
                        value = value["random"][0]
                    values.append(value)
        if _LIB.b13_set_preflop_strength((_D * 169)(*values)) != 169:
            raise ValueError("169 finite probabilities in [0,1] required")

    def five_card_histogram(self):
        counts = (C.c_uint64 * 9)()
        assert _LIB.b13_five_card_histogram(counts) == 2598960
        return tuple(counts)

    def icm_equity(self, stacks, payouts, hero=0):
        """Exact caller-supplied-payout ICM for one player, including zero stacks.

        Zero-chip players split unresolved trailing rank equities uniformly.
        This does not assign eliminated players a known finishing order; callers
        with known bust order should remove them and adjust payouts accordingly.
        """
        n = len(stacks)
        if not 1 <= n <= 8 or len(payouts) < n or not 0 <= hero < n:
            raise ValueError("1..8 stacks, matching payouts, valid hero required")
        value = _LIB.b13_icm_equity((_D*n)(*stacks), (_D*n)(*payouts[:n]), n, hero)
        if not math.isfinite(value):
            raise ValueError("finite nonnegative stacks and finite payouts required")
        return value


def combo_index(a, b):
    if not 0 <= a < b < 52:
        raise ValueError("combo_index requires 0 <= a < b < 52")
    return a * (103-a) // 2 + b-a-1
