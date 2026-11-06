"""Create an offline runtime ZIP and manifest, excluding control/private data."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    runtime = [Path('agent.py'), Path('README.md')]
    runtime += sorted(Path('pokerbot').glob('*.py'))
    runtime += [Path('assets/equity.bin.z'), Path('assets/preflop.json')]
    manifest = {str(p).replace('\\','/'): {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in runtime}
    Path('benchmarks/runtime_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    paths = runtime + [Path('RULES.txt')]
    paths += sorted(Path('tools').glob('*.py')) + sorted(Path('tools').glob('*.cpp'))
    paths += sorted(Path('tests').glob('*.py')) + sorted(Path('benchmarks').glob('*.md'))
    paths += [p for p in sorted(Path('benchmarks').glob('*.json')) if not p.name.startswith('package_validation')]
    paths += [Path('benchmarks/final_equity_test.bin')]
    with zipfile.ZipFile('bot_11_offline.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, arcname=p.as_posix())
    size = Path('bot_11_offline.zip').stat().st_size
    if size > 100_000_000:
        raise RuntimeError('package exceeds conservative 100MB limit')
    print(json.dumps({'archive':'bot_11_offline.zip','bytes':size,'sha256':hashlib.sha256(Path('bot_11_offline.zip').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
