"""Measure actual full act() wall and CPU time; never hides overruns.

GC remains enabled. Constructor and asset loading separately reported.
Collect observations from distinct tables, then replay with fresh table state.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from benchmarks.engine import play_table
from benchmarks.opponents import BaselineAgent
from bot.agent import Agent
from bot.protocol import validate_observation


def distribution(ns):
    ns = sorted(ns)
    def pct(q):
        return ns[min(len(ns) - 1, int((len(ns) - 1) * q))] / 1000
    return dict(n=len(ns), mean_us=statistics.fmean(ns) / 1000,
                p50_us=pct(.5), p95_us=pct(.95), p99_us=pct(.99),
                p999_us=pct(.999), max_us=ns[-1] / 1000,
                over_1ms=sum(x > 1000000 for x in ns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tables', type=int, default=80)
    p.add_argument('--repeat', type=int, default=4)
    p.add_argument('--output', default='artifacts/latency.json')
    args = p.parse_args()
    init = time.perf_counter_ns()
    Agent()
    init = time.perf_counter_ns() - init
    dataset = []
    seed = 731001
    for table in range(args.tables):
        seat = table % 8
        rows = []
        def observe(o, action, elapsed):
            if o['hero'] == seat:
                validate_observation(o)
                rows.append(o)
        opponents = [BaselineAgent(style, seed + table * 7 + i) for i, style in enumerate(['balanced', 'tight', 'calling', 'aggressive', 'shove', 'random', 'balanced'])]
        opponents.insert(seat, Agent(seed + table))
        play_table(opponents, seed + table, payouts=[50, 30, 20, 0, 0, 0, 0, 0], observer=observe)
        dataset.append(rows)
    wall, cpu, cold = [], [], []
    bystreet = {s: [] for s in range(4)}
    errors = 0
    for repeat in range(args.repeat):
        for table, rows in enumerate(dataset):
            agent = Agent(seed + table)
            for i, o in enumerate(rows):
                t0 = time.perf_counter_ns()
                c0 = time.process_time_ns()
                agent.act(o)
                c1 = time.process_time_ns()
                t1 = time.perf_counter_ns()
                wall.append(t1 - t0)
                cpu.append(c1 - c0)
                bystreet[o['street']].append(t1 - t0)
                if i == 0:
                    cold.append(t1 - t0)
            errors += agent.errors
    report = dict(platform=platform.platform(), python=platform.python_version(),
                  constructor_us=init / 1000, gc_enabled=True, tables=args.tables,
                  cpu_timer_note='On Windows GetProcessTimes may advance in ~15.625ms quanta despite its advertised resolution; per-decision CPU quantiles there are not useful. Wall-clock results are always reported.',
                  repeated_corpus_note='Repeated replay corpus, not independent performance samples.',
                  wall=distribution(wall), cpu=distribution(cpu),
                  first_decision=distribution(cold),
                  by_street={k: distribution(v) for k, v in bystreet.items() if v},
                  internal_error_fallbacks=errors)
    try:
        import resource
        report['process_peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        pass
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
