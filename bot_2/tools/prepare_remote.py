"""Prepare only this bot's sources/assets for the optional bounded Linux check."""
from pathlib import Path
import zipfile

files = [*Path("pokerbot").glob("*.py"), *Path("tools").glob("*.py"),
         *Path("tests").glob("*.py"), *Path("assets").glob("*.json"),
         *Path("assets").glob("*.jsonl"), Path("tools/remote_benchmark.sbatch")]
with zipfile.ZipFile(".control/bot_2_linux.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr("benchmarks/", "")
    for p in sorted(files):
        z.write(p, p.as_posix())
print("Prepared .control/bot_2_linux.zip")
