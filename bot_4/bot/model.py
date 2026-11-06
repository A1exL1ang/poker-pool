"""Small, deterministic, stdlib-only inference for offline-generated assets."""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent / "assets"
with (_ROOT / "preflop.json").open() as _f:
    PREFLOP = json.load(_f)
BASE = TREES = None


def predict_reference(x):
    global BASE, TREES
    if BASE is None:
        with (_ROOT / "equity.json").open() as f:
            model = json.load(f)
        BASE = tuple(model["base"])
        TREES = tuple(tuple(tuple(n) for n in t) for t in model["trees"])
    a, b, c, d, e, f, g, h = BASE
    for nodes in TREES:
        n = nodes[0]
        while n[0] >= 0:
            n = nodes[n[2] if x[n[0]] <= n[1] else n[3]]
        a += n[4]
        b += n[5]
        c += n[6]
        d += n[7]
        e += n[8]
        f += n[9]
        g += n[10]
        h += n[11]
    return tuple(min(1.0, max(0.0, v)) for v in (a, b, c, d, e, f, g, h))


# Only immutable model data are loaded at initialization. The generated source
# is our own model, checked against JSON inference by the test suite.
from .compiled_model import predict
