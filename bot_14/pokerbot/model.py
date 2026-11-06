"""Small deterministic inference engine for our offline equity surrogate."""
from __future__ import annotations
import json
from pathlib import Path
from .derived import augment


class EquityModel:
    def __init__(self, path=None):
        path = Path(path) if path else Path(__file__).resolve().parent.parent / "assets" / "equity_model.json"
        self.loaded = path.exists()
        raw = json.loads(path.read_text()) if self.loaded else {}
        self.models = raw.get("models", {})
        self.feature_count = raw.get("feature_count", 47)

    def predict(self, features, preflop_fraction=1., postflop_fraction=1., opponents=1, force_heuristic=False):
        if not self.loaded or force_heuristic:
            c, hi = features[1], features[2]
            # Development-only fallback, with range/texture conservatism.
            base = (.19 + .018 * features[7], .48 + .019 * hi, .80, .88, .92, .95, .98, .995, 1.)[c]
            if features[21] == 4 and len(features) and features[0] < 5:
                base += .10
            base -= .14 * (1 - preflop_fraction) + .22 * (1 - postflop_fraction)
            return max(.01, min(.999, base)) ** max(1., opponents * .7)
        q, m, n = max(.045, min(1., preflop_fraction)), max(.06, min(1., postflop_fraction)), max(1, min(7, opponents))
        x = augment(features, q, m, n) if self.feature_count > 47 else tuple(features) + (q, m, n)
        model = self.models[str(features[0])]
        value = model["bias"]
        for tree in model["trees"]:
            node = tree[0]
            while node[0] >= 0:
                node = tree[node[2] if x[node[0]] <= node[1] else node[3]]
            value += node[1]
        return max(.002, min(.998, value))


MODEL = EquityModel()
