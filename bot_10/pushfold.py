"""Constant-time interpolation of the documented offline first-caller model."""
import json
from pathlib import Path

_DATA=json.loads((Path(__file__).resolve().parent/'assets/pushfold.json').read_text())
_INDEX={key:i for i,key in enumerate(_DATA['keys'])}
_DEPTHS=_DATA['metadata']['depths_bb']


def probability(key, depth, behind, caller=None):
    """Return a mixed frequency. caller=None means shove, otherwise call index."""
    behind=max(1,min(7,behind));i=_INDEX[key]
    low=max((d for d in _DEPTHS if d<=depth),default=_DEPTHS[0])
    high=min((d for d in _DEPTHS if d>=depth),default=_DEPTHS[-1])
    def value(d):
        if caller is None: return _DATA['shove'][str(d)][str(behind)][i]
        return _DATA['call'][str(d)][str(behind)][caller][i]
    q=value(low)
    return q if high==low else q+(value(high)-q)*(depth-low)/(high-low)
