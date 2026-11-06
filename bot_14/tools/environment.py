"""Small environment check; safe to run on the login node (no computation)."""
import importlib.util
import json
import sys
print(json.dumps({"python": sys.version, "packages": {m: bool(importlib.util.find_spec(m)) for m in ("numpy", "sklearn", "scipy")}}))
