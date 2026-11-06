"""Fingerprint only this bot's runtime source and assets."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def runtime_files():
    return sorted([ROOT / "agent.py", *list((ROOT / "bot7").glob("*.py")),
                   ROOT / "assets" / "eval.bin.gz", ROOT / "assets" / "preflop.json"])


def fingerprint():
    files = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_files()}
    digest = hashlib.sha256("\n".join(f"{k}:{v}" for k, v in files.items()).encode()).hexdigest()
    return {"sha256": digest, "files": files}
