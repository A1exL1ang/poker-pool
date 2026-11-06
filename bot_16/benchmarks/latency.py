"""Full Agent.act wall/CPU latency benchmark; initialization reported separately.

Run after freezing code/assets, without concurrent CPU-heavy experiments:
  python benchmarks/latency.py --observations benchmarks/observations.json --output benchmarks/latency.json
Synthetic fixtures are local contract tests, not official timing guarantees.
"""
import argparse
import copy
from collections import Counter
import importlib
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from time import perf_counter_ns, process_time_ns, thread_time_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import Baseline
from simulator import play_hand
from benchmarks.tournament import fingerprints


def action_error(action, obs):
    """Independent contract validation, outside timed section."""
    if not isinstance(action, dict):
        return 'not a dictionary'
    kind = action.get('action')
    if kind not in ('fold', 'check', 'call', 'raise'):
        return 'unknown action'
    call = obs['to_call']
    if kind == 'check' and call:
        return 'check facing wager'
    if kind == 'call' and not call:
        return 'call without wager'
    if kind == 'raise':
        target = action.get('amount')
        p = obs['players'][obs['hero']]
        if type(target) is not int:
            return 'non-integer raise total'
        if not obs['can_raise'] or target <= p['street_bet'] + call or target > obs['max_raise_to']:
            return 'raise unavailable or outside stack'
        if target < obs['min_raise_to'] and target != obs['max_raise_to']:
            return 'non-all-in underraise'
    return None


def random_observations(hands=128, seed=160900):
    """Randomly dealt, legal histories produced independently by local engine."""
    rng = random.Random(seed)
    by_street = {i: [] for i in range(4)}
    styles = ('tag', 'lag', 'station', 'maniac', 'random')
    for hand in range(hands):
        n = rng.randint(2, 8)
        agents = [Baseline(rng.choice(styles), seed + hand * 8 + i) for i in range(n)]
        stacks = [rng.randint(40, 3000) for _ in range(n)]
        def collect(seat, obs, action, duration):
            by_street[obs['street']].append(obs)
        play_hand(agents, stacks, seed=seed + hand, hand_id=hand,
                  button=hand % n, observe=collect,
                  payouts=[50, 30, 20, 0, 0, 0, 0, 0][:n])
    out = []
    for street, observations in by_street.items():
        rng.shuffle(observations)
        # Balance streets so numerous cheap preflop decisions do not conceal
        # postflop costs. This distribution is intentionally not natural play.
        for obs in observations[:min(512, len(observations))]:
            obs = dict(obs)
            obs['hand_id'] = 1_000_000 + len(out)
            out.append(obs)
    rng.shuffle(out)
    return out


def stress_observations():
    fixtures = {}
    # Eight tournament survivors, six folded hands, one heads-up all-in call:
    # expensive payout-aware path. Contributions correspond to limping then
    # facing a BB shove; payout vector is explicitly synthetic.
    for street, board in ((0, []), (1, [0, 17, 30]), (2, [0, 17, 30, 43]), (3, [0, 17, 30, 43, 9])):
        players = [dict(stack=1980, street_bet=20 if street == 0 else 0,
                        invested=20, folded=i >= 2, all_in=False) for i in range(8)]
        players[1] = dict(stack=0, street_bet=2000 if street == 0 else 1980,
                          invested=2000, folded=False, all_in=True)
        history = []
        for i in range(2, 8):
            history.append(dict(seat=i, street=0, action='call', paid=20, to=20, pot_before=30 + (i - 2) * 20))
        history.append(dict(seat=0, street=0, action='call', paid=10, to=20, pot_before=150))
        if street:
            history.append(dict(seat=1, street=0, action='check', paid=0, to=20, pot_before=160))
            for earlier in range(1, street):
                for i in range(8):
                    history.append(dict(seat=i, street=earlier, action='check', paid=0, to=0, pot_before=160))
            history.append(dict(seat=0, street=street, action='check', paid=0, to=0, pot_before=160))
        history.append(dict(seat=1, street=street, action='raise', paid=1980,
                            to=players[1]['street_bet'], pot_before=160))
        for i in range(2, 8):
            history.append(dict(seat=i, street=street, action='fold', paid=0,
                                to=players[i]['street_bet'], pot_before=2140))
        fixtures[f'allin_icm8_street{street}'] = dict(
            hero=0, button=7, hand_id=2_000_000 + street, street=street,
            hole=[48, 49], board=board, players=players, pot=2140,
            to_call=1980, big_blind=20, small_blind=10,
            min_raise_to=3980 if street == 0 else 3960,
            max_raise_to=2000 if street == 0 else 1980, can_raise=False,
            history=history, payouts=[50, 30, 20, 0, 0, 0, 0, 0])
    # Covered all-in calls keep all eight tournament stacks alive in both the
    # fold and win branches, exercising the largest ICM DP states. Redistribute
    # 1000 chips from folded stacks to the bettor to preserve the 16000 total.
    for street in range(4):
        covered = copy.deepcopy(fixtures[f'allin_icm8_street{street}'])
        covered['players'][1]['stack'] = 1000
        covered['players'][1]['all_in'] = False
        for seat in range(2, 8):
            covered['players'][seat]['stack'] -= 167 if seat < 6 else 166
        fixtures[f'allin_icm8_covered_street{street}'] = covered
    # Legally generate long histories. Two alternating aggressors minraise,
    # other seats call, until 98 raises (the last full-raise level leaving chips at 100BB); all then check to showdown. This
    # reaches 8-way postflop inference with a substantial preflop history.
    class LongHistory:
        def act(self, obs):
            raises = sum(a['action'] == 'raise' for a in obs['history'])
            if obs['street'] == 0 and obs['hero'] in (0, 4) and raises < 98 and obs['can_raise']:
                return {'action': 'raise', 'amount': min(obs['min_raise_to'], obs['max_raise_to'])}
            return {'action': 'call' if obs['to_call'] else 'check'}
    longest = {}
    def collect(seat, obs, action, duration):
        street = obs['street']
        if street not in longest or len(obs['history']) > len(longest[street]['history']):
            longest[street] = obs
    result = play_hand([LongHistory() for _ in range(8)], [2000] * 8,
                       seed=160901, button=7, hand_id=2_100_000, observe=collect)
    if result.violations:
        raise AssertionError(result.violations)
    for street, obs in longest.items():
        fixtures[f'legal_long_history_street{street}'] = obs
    return fixtures


def load_observations(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError('Observation file must be within this workspace')
    payload = json.loads(resolved.read_text(encoding='utf-8-sig'))
    rows = payload.get('observations') if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows or not all(isinstance(o, dict) for o in rows):
        raise ValueError('Expected nonempty list of observation dictionaries')
    return rows


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0)
    def p(q):
        return values[min(len(values) - 1, int((len(values) - 1) * q))] / 1000
    return dict(count=len(values), p50_us=p(.50), p95_us=p(.95), p99_us=p(.99),
                p999_us=p(.999), max_us=values[-1] / 1000,
                mean_us=statistics.mean(values) / 1000,
                overruns_1ms=sum(v > 1_000_000 for v in values))


def measure(agent, observations, repeats, force_history_parse=False):
    walls, cpus, thread_cpus, errors = [], [], [], []
    action_counts = Counter()
    initial_errors = getattr(agent, 'errors', 0)
    initial_fallbacks = getattr(agent, 'fallbacks', 0)
    sequence = 0
    for repeat in range(repeats):
        for original in observations:
            obs = original
            if force_history_parse:
                obs = dict(original)
                obs['hand_id'] = 9_000_000 + sequence
            c0 = process_time_ns()
            t0 = thread_time_ns()
            w0 = perf_counter_ns()
            try:
                action = agent.act(obs)
            except Exception as exc:
                action = None
                exception = f'{type(exc).__name__}: {exc}'
            else:
                exception = None
            w1 = perf_counter_ns()
            t1 = thread_time_ns()
            c1 = process_time_ns()
            walls.append(w1 - w0)
            cpus.append(c1 - c0)
            thread_cpus.append(t1 - t0)
            error = exception or action_error(action, obs)
            if error:
                errors.append(dict(index=sequence, error=error))
            elif action:
                action_counts[f"{obs['street']}:{action['action']}"] += 1
            sequence += 1
    process_positive = min((v for v in cpus if v > 0), default=0)
    thread_positive = min((v for v in thread_cpus if v > 0), default=0)
    return dict(wall=distribution(walls), process_cpu=distribution(cpus),
                thread_cpu=distribution(thread_cpus),
                process_cpu_sub_ms_resolution_observed=0 < process_positive < 1_000_000,
                thread_cpu_sub_ms_resolution_observed=0 < thread_positive < 1_000_000,
                thread_cpu_zero_fraction=sum(v == 0 for v in thread_cpus) / len(thread_cpus),
                thread_cpu_smallest_positive_us=thread_positive / 1000,
                illegal_or_raised_error_count=len(errors), first_errors=errors[:10],
                internal_errors=getattr(agent, 'errors', 0) - initial_errors,
                internal_fallbacks=getattr(agent, 'fallbacks', 0) - initial_fallbacks,
                cpu_zero_fraction=sum(v == 0 for v in cpus) / len(cpus),
                cpu_smallest_positive_us=min((v for v in cpus if v > 0), default=0) / 1000,
                action_counts=dict(action_counts), observations=len(observations), repeats=repeats,
                force_full_history_parse=force_history_parse,
                max_history_entries=max(len(o.get('history', [])) for o in observations),
                max_live_hand_opponents=max(sum(not p['folded'] for p in o['players']) - 1 for o in observations))


def run(observation_path=None, repeats=5, random_hands=128, stress_repeats=200,
        seed=160900, require_assets=True):
    hashes = fingerprints()
    fixtures = stress_observations()
    random_rows = random_observations(random_hands, seed)
    observed = load_observations(observation_path) if observation_path else None
    import_start = perf_counter_ns()
    Agent = importlib.import_module('agent').Agent
    equity_module = importlib.import_module('bot.equity')
    import_ns = perf_counter_ns() - import_start
    available = dict(model=equity_module.MODEL_AVAILABLE,
                     preflop=equity_module.PREFLOP_AVAILABLE,
                     narrow=getattr(equity_module, '_NARROW', None) is not None,
                     compiled=getattr(equity_module, 'COMPILED_AVAILABLE', False))
    if require_assets and not all(available.values()):
        raise RuntimeError('Final latency run requires all three trained assets: ' + repr(available))
    init_start = perf_counter_ns()
    cold_agent = Agent(seed=seed)
    constructor_ns = perf_counter_ns() - init_start
    # This is the first full act in this fresh process, deliberately using the
    # expensive eight-survivor all-in ICM path. Import is outside the deadline.
    cold = measure(cold_agent, [fixtures['allin_icm8_street0']], 1)
    workloads = {}
    if observed:
        workloads['saved_tournament_observations'] = measure(Agent(seed=seed), observed, repeats)
    workloads['random_legal_street_balanced'] = measure(Agent(seed=seed), random_rows, repeats)
    for name, observation in fixtures.items():
        workloads[name] = measure(Agent(seed=seed), [observation], stress_repeats, force_history_parse=True)
        print(name, 'p99 wall us', workloads[name]['wall']['p99_us'],
              'wall overruns', workloads[name]['wall']['overruns_1ms'], flush=True)
    measured = [cold] + list(workloads.values())
    integrity_ok = all(not m['illegal_or_raised_error_count'] and not m['internal_errors'] and not m['internal_fallbacks'] for m in measured)
    total_calls = sum(m['wall']['count'] for m in measured)
    total_overruns = sum(m['wall']['overruns_1ms'] for m in measured)
    return dict(purpose='full Agent.act latency including features, inference, ICM and action normalization',
                action_integrity_ok=integrity_ok, total_decision_calls=total_calls,
                total_wall_overruns_1ms=total_overruns,
                wall_target_met_for_every_measured_call=total_overruns == 0,
                python=sys.version, platform=platform.platform(), seed=seed,
                timing_scope='warm imports; clocks surround full act only; CPU counters also include clock-read overhead',
                clock_resolution_seconds={name: time.get_clock_info(name).resolution for name in ('perf_counter', 'process_time', 'thread_time')},
                cpu_timing_caveat='Windows process/thread CPU accounting may advance in ~15.625ms ticks despite reported clock resolution; per-decision CPU quantiles/overruns then reflect quantization. Use Linux for meaningful sub-ms CPU distributions.',
                official_hardware_unknown=True, gc_policy='Python default; not disabled',
                assets_available=available, file_sha256=hashes,
                files_changed_during_run=fingerprints() != hashes,
                import_ms=import_ns / 1e6, constructor_us=constructor_ns / 1000,
                cold_first_action=cold, cold_fixture='allin_icm8_street0',
                saved_observation_file=str(observation_path) if observation_path else None,
                workloads=workloads,
                note='Wall scheduling pauses are retained. No observed overrun is discarded. Synthetic stress frequencies are not tournament frequencies.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--random-hands', type=int, default=128)
    p.add_argument('--stress-repeats', type=int, default=200)
    p.add_argument('--seed', type=int, default=160900)
    p.add_argument('--allow-development-assets', action='store_true')
    p.add_argument('--output', default='benchmarks/latency.json')
    a = p.parse_args()
    if min(a.repeats, a.random_hands, a.stress_repeats) < 1:
        p.error('workload sizes must be positive')
    report = run(a.observations, a.repeats, a.random_hands, a.stress_repeats,
                 a.seed, not a.allow_development_assets)
    Path(a.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('saved', a.output, 'source changed:', report['files_changed_during_run'],
          'wall overruns:', report['total_wall_overruns_1ms'], '/', report['total_decision_calls'])
    if not report['action_integrity_ok'] or report['files_changed_during_run']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
