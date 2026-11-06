"""Optional packaged native equity backend, with a portable Python fallback.

No compilation, subprocess, installation or network operation occurs here.
An unavailable/incompatible binary simply leaves ``available`` false. The
caller chooses whether to use native rollouts; exact HU river enumeration is
the intended low-variance default. Multiway IID river estimates are explicitly
approximate and available only through a separately named method.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
import sys


class NativeEquity:
    def __init__(self, percentiles=None, path=None):
        self.available = False
        self.error = None
        self.calls = 0
        self.fallbacks = 0
        self.library = None
        self.model_feature_count = 0
        self.model_streets = set()
        self.model_error = None
        root = Path(__file__).resolve().parent
        if path is None:
            filename = "b14_equity.dll" if sys.platform == "win32" else "libb14_equity.so" if sys.platform.startswith("linux") else None
            if filename is None:
                self.error = "no packaged backend for this platform"
                return
            candidates = (root / "_native" / filename, root / filename, root.parent / "assets" / "native" / filename)
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                self.error = "optional native binary absent"
                return
        try:
            library = ctypes.CDLL(str(Path(path).resolve()))
            library.b14_version.argtypes = ()
            library.b14_version.restype = ctypes.c_uint
            if library.b14_version() != 1:
                raise ValueError("native equity ABI version mismatch")
            library.b14_init_ranks.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_int)
            library.b14_init_ranks.restype = ctypes.c_int
            library.b14_equity.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                            ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_uint64, ctypes.c_int)
            library.b14_equity.restype = ctypes.c_double
            library.b14_river_iid.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                             ctypes.c_double, ctypes.c_double, ctypes.c_int)
            library.b14_river_iid.restype = ctypes.c_double
            library.b14_river_equities.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                                  ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                                  ctypes.POINTER(ctypes.c_double))
            library.b14_river_equities.restype = ctypes.c_int
            for name in ("b14_last_status", "b14_last_trials", "b14_last_fallbacks"):
                function = getattr(library, name)
                function.argtypes = ()
                function.restype = ctypes.c_int
            if percentiles is None:
                from .preflop import TABLES
                percentiles = TABLES.percentile
            if len(percentiles) != 169 or not all(math.isfinite(v) and 0 <= v <= 1 for v in percentiles):
                raise ValueError("need 169 finite preflop class percentiles")
            values = (ctypes.c_double * 169)(*percentiles)
            if not library.b14_init_ranks(values, 169):
                raise ValueError("native percentile initialization failed")
            self.library = library
            self.path = str(Path(path).resolve())
            self.available = True
        except (OSError, AttributeError, ValueError, TypeError) as exc:
            self.error = str(exc)

    def configure_model(self, models, feature_count=47):
        """Copy validated offline trees into native startup-only storage.

        ``models`` is the JSON model's street mapping. This is an optional ABI
        extension: older binaries keep their equity functions and return
        False here, allowing normal Python model inference.
        """
        self.model_streets = set()
        self.model_feature_count = 0
        self.model_error = None
        if not self.available:
            self.model_error = "native backend unavailable"
            return False
        try:
            init = self.library.b14_init_model
            predict = self.library.b14_predict_model
            init.argtypes = (ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
                             ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int)
            init.restype = ctypes.c_int
            predict.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int)
            predict.restype = ctypes.c_double
            if type(feature_count) is not int or not 1 <= feature_count <= 128:
                raise ValueError("unsupported native model feature count")
            flattened = []
            for street in (3, 4, 5):
                model = models.get(str(street), models.get(street))
                if model is None:
                    continue
                bias = float(model["bias"])
                trees = model["trees"]
                if not math.isfinite(bias) or abs(bias) > 1e6 or len(trees) > 2048:
                    raise ValueError("invalid model bias or tree count")
                features, thresholds, left, right, leaves, roots = [], [], [], [], [], []
                for tree in trees:
                    if not tree:
                        raise ValueError("empty model tree")
                    colors = [0] * len(tree)

                    def visit(index, depth=0):
                        if type(index) is not int or not 0 <= index < len(tree) or depth > 128:
                            raise ValueError("invalid model child index or depth")
                        if colors[index] == 1:
                            raise ValueError("cycle in model tree")
                        if colors[index] == 2:
                            return
                        colors[index] = 1
                        node = tree[index]
                        if not isinstance(node, (tuple, list)) or len(node) not in (2, 4):
                            raise ValueError("invalid model node")
                        feature = node[0]
                        if type(feature) is not int or feature < -1 or feature >= feature_count or not math.isfinite(node[1]):
                            raise ValueError("invalid model feature or threshold")
                        if feature >= 0:
                            if len(node) != 4:
                                raise ValueError("branch node requires two children")
                            visit(node[2], depth + 1)
                            visit(node[3], depth + 1)
                        elif len(node) != 2 or abs(node[1]) > 1e6:
                            raise ValueError("invalid model leaf")
                        colors[index] = 2

                    visit(0)
                    offset = len(features)
                    roots.append(offset)
                    for node in tree:
                        feature = node[0]
                        if feature >= 0:
                            features.append(feature); thresholds.append(float(node[1]))
                            left.append(offset + node[2]); right.append(offset + node[3]); leaves.append(0.0)
                        else:
                            features.append(-1); thresholds.append(0.0)
                            left.append(-1); right.append(-1); leaves.append(float(node[1]))
                if len(features) > 131072:
                    raise ValueError("native model node capacity exceeded")
                flattened.append((street, bias, features, thresholds, left, right, leaves, roots))
            if not flattened:
                raise ValueError("no street models configured")
            for street, bias, features, thresholds, left, right, leaves, roots in flattened:
                count = len(features)
                c_features = (ctypes.c_int * count)(*features)
                c_thresholds = (ctypes.c_double * count)(*thresholds)
                c_left = (ctypes.c_int * count)(*left)
                c_right = (ctypes.c_int * count)(*right)
                c_leaves = (ctypes.c_double * count)(*leaves)
                c_roots = (ctypes.c_int * len(roots))(*roots)
                if not init(street, bias, c_features, c_thresholds, c_left, c_right, c_leaves, count, c_roots, len(roots), feature_count):
                    raise ValueError("native model initialization rejected tree arrays")
                self.model_streets.add(street)
            self.model_feature_count = feature_count
            self._predict_model = predict
            return True
        except (AttributeError, KeyError, ValueError, TypeError, IndexError, OverflowError, ctypes.ArgumentError) as exc:
            self.model_streets = set()
            self.model_feature_count = 0
            self.model_error = str(exc)
            return False

    def predict_model(self, features, q=1.0, m=1.0, opponents=1):
        """Native surrogate prediction from raw44 or already augmented input."""
        if not self.model_feature_count:
            return None
        try:
            street = int(features[0])
            if street not in self.model_streets:
                return None
            if len(features) == self.model_feature_count:
                vector = features
            elif len(features) == 44:
                q, m, opponents = max(.045, min(1., q)), max(.06, min(1., m)), max(1, min(7, opponents))
                if self.model_feature_count > 47:
                    from .derived import augment
                    vector = augment(features, q, m, opponents)
                else:
                    vector = tuple(features) + (q, m, opponents)
            else:
                return None
            if len(vector) != self.model_feature_count:
                return None
            values = (ctypes.c_double * len(vector))(*vector)
            result = self._predict_model(street, values, len(vector))
            return result if math.isfinite(result) and 0 <= result <= 1 else None
        except (ValueError, TypeError, IndexError, OverflowError, ctypes.ArgumentError):
            return None

    def estimate(self, hole, board, opponents=1, q=1.0, m=1.0, seed=14, samples=256):
        """Exact HU river share, or bounded conditional Monte Carlo elsewhere.

        Returns ``None`` for unavailable backend, bad input or exhausted joint
        rejection budget. Its caller then uses the portable equity model.
        """
        self.calls += 1
        if not self.available:
            self.fallbacks += 1
            return None
        try:
            values = (ctypes.c_int * len(board))(*board)
            value = self.library.b14_equity(hole[0], hole[1], values, len(board), q, m, opponents, seed, samples)
            if not math.isfinite(value) or not 0 <= value <= 1:
                self.fallbacks += 1
                return None
            return value
        except (ValueError, TypeError, IndexError, ctypes.ArgumentError):
            self.fallbacks += 1
            return None

    def river_iid(self, hole, board, opponents=2, q=1.0, m=1.0):
        """Approximate multiway river equity, ignoring card correlation."""
        if not self.available:
            return None
        try:
            values = (ctypes.c_int * len(board))(*board)
            result = self.library.b14_river_iid(hole[0], hole[1], values, len(board), q, m, opponents)
            return result if math.isfinite(result) and 0 <= result <= 1 else None
        except (ValueError, TypeError, IndexError, ctypes.ArgumentError):
            return None

    def river_equities(self, hole, board, q=1.0, m_values=(1.0,)):
        """Exact HU river shares for several m cutoffs with shared sorting."""
        if not self.available or not 1 <= len(m_values) <= 8:
            return None
        try:
            values = (ctypes.c_int * len(board))(*board)
            cutoffs = (ctypes.c_double * len(m_values))(*m_values)
            output = (ctypes.c_double * len(m_values))()
            success = self.library.b14_river_equities(hole[0], hole[1], values, len(board), q, cutoffs, len(m_values), output)
            return tuple(output) if success else None
        except (ValueError, TypeError, IndexError, ctypes.ArgumentError):
            return None

    def diagnostics(self):
        if not self.available:
            return {"available": False, "error": self.error, "calls": self.calls, "fallbacks": self.fallbacks}
        return {"available": True, "calls": self.calls, "fallbacks": self.fallbacks,
                "status": self.library.b14_last_status(), "trials": self.library.b14_last_trials(),
                "rejected_deals": self.library.b14_last_fallbacks()}
