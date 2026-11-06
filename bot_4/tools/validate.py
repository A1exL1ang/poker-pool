"""Reproduce final checks and measurements with one bounded local command."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='2,300 tournaments plus exhaustive evaluator census')
    args = ap.parse_args()
    start = time.perf_counter()
    commands = []

    def run(arguments, log):
        print('Running:', ' '.join(arguments), flush=True)
        commands.append([sys.executable, *arguments])
        with (ROOT / 'evidence' / log).open('w') as f:
            subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)

    run(['-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'tests.txt')
    if args.full:
        run(['tools/evaluator_census.py'], 'evaluator_census.log')
    run(['tools/calibrate.py', '--states', '800' if args.full else '100'], 'exact_river_calibration.log')
    fields = [('mixed',500,81000,'top3'),('equity',300,91000,'top3'),('tight',300,101000,'top3'),
              ('station',300,111000,'top3'),('self',400,131000,'top3'),('mixed',500,121000,'wta')]
    for field, n, seed, payout in fields:
        name = 'wta' if payout == 'wta' else field
        command = ['tools/simulate.py','--tables',str(n if args.full else 20),'--seed',str(seed),
                   '--field',field,'--payout',payout,'--output',f'evidence/{name}_final.json']
        if field == 'mixed' and payout == 'top3':
            command.append('--capture')
        run(command, f'{name}_final.log')
    # Run timing last with no trainer or simulator child processes active.
    run(['tools/benchmark.py','--rounds','10' if args.full else '3','--output','evidence/latency_final.json'], 'latency_final.log')
    source = ['agent.py', *[str(p.relative_to(ROOT)).replace('\\','/') for p in sorted((ROOT/'bot').glob('*.py'))],
              'assets/preflop.json','assets/equity.json']
    report = {'full': args.full, 'python':sys.version, 'platform':platform.platform(),
              'seconds':time.perf_counter()-start, 'commands':commands,
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source},
              'passed':True}
    (ROOT/'evidence/validation_run.json').write_text(json.dumps(report,indent=2))
    print(f'Validation completed in {report["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
