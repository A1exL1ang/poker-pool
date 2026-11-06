"""Seeded rule/observation integration fuzzing; not strategic held-out evaluation."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from benchmarks.engine import play_table
from benchmarks.opponents import BaselineAgent
from bot.agent import Agent
from bot.protocol import validate_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=7310001)
    parser.add_argument('--output', default='artifacts/integration_fuzz.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ('bot/agent.py', 'bot/equity.py', 'benchmarks/engine.py')}
    model = (root / 'artifacts/equity_model.json.gz').exists()
    start = time.time()
    totals = dict(table_decisions=0, engine_errors=0, invalid_actions=0,
                  candidate_decisions=0, candidate_internal_fallbacks=0,
                  candidate_wall_clock_over_1ms=0, max_candidate_ns=0)
    styles = ['tight', 'calling', 'random', 'aggressive', 'shove', 'balanced', 'calling']
    # Payout vectors below are declared test cases only, not official scoring.
    payouts = [None, [50, 30, 20, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1] * 8]
    for k in range(args.tables):
        seed = args.seed + k
        seat = k % 8
        agent = Agent(seed=seed)
        agents = [BaselineAgent(styles[(i + k) % 7], seed=seed * 17 + i) for i in range(7)]
        agents.insert(seat, agent)

        def observe(obs, action, elapsed):
            validate_observation(obs)
            if obs['hero'] == seat:
                totals['candidate_decisions'] += 1
                totals['candidate_wall_clock_over_1ms'] += elapsed > 1000000
                totals['max_candidate_ns'] = max(totals['max_candidate_ns'], elapsed)
                assert isinstance(action, dict)
                assert type(action['amount']) is int
                if action['action'] == 'call':
                    assert action['amount'] == min(obs['to_call'], obs['stacks'][seat])
                if action['action'] in ('fold', 'check'):
                    assert action['amount'] == 0

        result = play_table(agents, seed, payouts=payouts[k % len(payouts)], observer=observe)
        assert sum(result['stacks']) == 16000
        assert 1 <= result['hands'] <= 200
        totals['table_decisions'] += result['decisions']
        totals['engine_errors'] += result['errors']
        totals['invalid_actions'] += result['invalid_actions']
        totals['candidate_internal_fallbacks'] += agent.errors
        if (k + 1) % 100 == 0:
            print(json.dumps(dict(completed=k + 1, elapsed_seconds=round(time.time() - start, 2), **totals)), flush=True)
    result = dict(config=vars(args), source_sha256=hashes,
                  trained_model_present_at_import=model, seconds=time.time() - start,
                  purpose='integration fuzz; not held-out strategy evaluation or final latency certification',
                  timing='wall clock, unenforced, includes host scheduling; all payouts are synthetic fixtures',
                  totals=totals)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
