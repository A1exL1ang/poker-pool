"""Explicit offline-only model selection for reproducing earlier experiments."""
import json
from pathlib import Path


def load_model(path):
    import equity
    trees=json.loads(Path(path).read_text())['trees']
    equity.TREES=[(t['left'],t['right'],t['feature'],t['threshold'],t['value']) for t in trees]
