"""Small offline-trained equity forest. No sampling or external calls in play."""
import json
from pathlib import Path
from features import features

_ROOT = Path(__file__).resolve().parent
PREFLOP = json.loads((_ROOT / 'assets/preflop_equity.json').read_text())['equities']
_model_path = _ROOT / 'assets/equity_model.json'
TREES = json.loads(_model_path.read_text())['trees'] if _model_path.exists() else []


def estimate(hole, board):
    """Equities vs uniform / heuristic top60 / top30 / top12 percent ranges."""
    f = features(hole, board)
    if not TREES:
        # Explicit development fallback, not used in packaged best candidate.
        cat = f[36]
        q = (.25, .65, .80, .88, .93, .96, .985, .997, .999)[cat]
        return [q, max(0, q-.04), max(0, q-.09), max(0, q-.15)]
    q = [0., 0., 0., 0.]
    for left, right, feat, threshold, values in TREES:
        i = 0
        while left[i] != -1:
            i = left[i] if f[feat[i]] <= threshold[i] else right[i]
        v = values[i]
        q[0] += v[0]; q[1] += v[1]; q[2] += v[2]; q[3] += v[3]
    n = len(TREES)
    return [x / n for x in q]


if TREES:
    TREES = [(t['left'], t['right'], t['feature'], t['threshold'], t['value']) for t in TREES]
