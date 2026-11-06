"""Dependency-free inference; asset loading happens at construction, never per act."""
import json
from pathlib import Path

from .cards import features, hand_class

ASSETS = Path(__file__).resolve().parent.parent / "assets"


class EquityModel:
    def __init__(self, assets=ASSETS):
        with (Path(assets) / "preflop.json").open(encoding="utf-8") as f:
            pf = json.load(f)
        self.preflop_table = pf["equity"]
        self.percentile = pf["percentile"]
        self.widths = pf["widths"]
        with (Path(assets) / "pushfold.json").open(encoding="utf-8") as f:
            self.pushfold = json.load(f)
        from ._equity_fast import predict
        self.predict_raw = predict

    def preflop(self, hole, width=1.0):
        row = self.preflop_table[hand_class(*hole)]
        if width >= 1:
            return row[0]
        if width <= self.widths[-1]:
            return row[-1]
        for i in range(1, len(self.widths)):
            if width >= self.widths[i]:
                w = (width - self.widths[i]) / (self.widths[i - 1] - self.widths[i])
                return row[i] * (1 - w) + row[i - 1] * w
        return row[-1]

    def predict_features(self, x):
        return tuple(max(0.001, min(0.999, z)) for z in self.predict_raw(x))

    def postflop(self, hole, board):
        x = features(hole, board)
        return self.predict_features(x), x

    def heads_up(self, hole, depth, action):
        """Interpolated mixed strategy from the offline push/fold subgame."""
        depths = self.pushfold["depths"]
        key = hand_class(*hole)
        rows = self.pushfold[action]
        if depth <= depths[0]:
            return rows[0][key]
        for i in range(1, len(depths)):
            if depth <= depths[i]:
                w = (depth - depths[i - 1]) / (depths[i] - depths[i - 1])
                return rows[i - 1][key] * (1 - w) + rows[i][key] * w
        return rows[-1][key]
