"""Bounded randomized mechanics audit against independent settlement/reference.

python tools/stress_engine.py --hands 2000 --out measurements/engine_properties.json
This uses only synthetic random actions, never opponent/private contest data.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import Hand, evaluate_reference
from evaluator import evaluate


def enumerated_awards(hand):
    """Independently bucket individual chip heights, then distribute each pot."""
    pots = {}
    for height in range(1, max(hand.contributions)+1):
        contributors = tuple(i for i, value in enumerate(hand.contributions) if value >= height)
        pots[contributors] = pots.get(contributors, 0) + len(contributors)
    result = [0] * hand.n
    ranks = {i: evaluate_reference(hand.holes[i]+hand.board)
             for i in range(hand.n) if not hand.folded[i]}
    for contributors, amount in pots.items():
        if len(contributors) == 1:
            result[contributors[0]] += amount
            continue
        eligible = [i for i in contributors if not hand.folded[i]]
        high = max(ranks[i] for i in eligible)
        winners = [i for i in eligible if ranks[i] == high]
        winners.sort(key=lambda i: (i-hand.button-1) % hand.n)
        for j, i in enumerate(winners):
            result[i] += amount//len(winners) + (j < amount % len(winners))
    return result


def audit(hands=2000, seed=9077):
    rng = random.Random(seed)
    decisions = showdowns = short_raises = full_raises = folds = 0
    started = time.perf_counter()
    for trial in range(hands):
        n = rng.randint(2, 8)
        stacks = [rng.randint(1, 300) if rng.random() > .15 else 0 for _ in range(n)]
        stacks[0] = max(1, stacks[0])
        stacks[1] = max(1, stacks[1])
        button = rng.choice([i for i in range(n) if stacks[i] > 0])
        sb = rng.randint(1, 30)
        deck = list(range(52))
        rng.shuffle(deck)
        suits = list(range(4))
        rng.shuffle(suits)
        remap = lambda c: c//4*4+suits[c % 4]
        hand = Hand(stacks, button, sb, 2*sb, deck=deck, evaluator=evaluate)
        permuted = Hand(stacks, button, sb, 2*sb, deck=list(map(remap, deck)), evaluator=evaluate)
        while not hand.done:
            obs = hand.observation()
            assert obs['hole'] == hand.holes[hand.actor]
            assert len(obs['hole']) == 2
            assert not set(obs).intersection({'holes', 'deck', 'opponent_cards', 'opponent_holes'})
            assert obs['pot'] == sum(hand.contributions)
            assert 0 <= obs['to_call'] <= obs['stacks'][obs['hero']]
            assert all(set(entry) == {'seat', 'street', 'action', 'amount'} for entry in obs['history'])
            assert not hand.folded[hand.actor] and hand.stacks[hand.actor] > 0
            choice = rng.random()
            if obs['can_raise'] and choice < .5:
                minimum, maximum = obs['min_raise_to'], obs['max_raise_to']
                target = maximum if choice < .25 else min(minimum, maximum)
                if choice > .4 and minimum < maximum:
                    target = rng.randint(minimum, maximum)
                action = {'action': 'raise', 'amount': target}
                short_raises += target < minimum
                full_raises += target >= minimum
            elif obs['to_call'] and choice > .8:
                action = {'action': 'fold'}
                folds += 1
            else:
                action = {'action': 'call' if obs['to_call'] else 'check'}
            hand.step(action)
            permuted.step(action)
            decisions += 1
            assert hand.stacks == permuted.stacks
            assert hand.contributions == permuted.contributions
            assert hand.actor == permuted.actor and hand.done == permuted.done
            assert sum(hand.stacks)+(0 if hand.done else sum(hand.contributions)) == sum(stacks)
            assert all(s >= 0 and isinstance(s, int) for s in hand.stacks)
            visible = hand.board+[c for hole in hand.holes for c in hole]
            assert len(visible) == len(set(visible))
        if len(hand.board) == 5:
            showdowns += 1
            assert hand.awards == enumerated_awards(hand)
        assert hand.awards == permuted.awards
    return dict(audit='randomized-engine-differential-v1', hands=hands, seed=seed,
                decisions=decisions, showdowns=showdowns, full_raises=full_raises,
                short_all_in_raises=short_raises, folds=folds, failures=0,
                checks=['per-action chip conservation and nonnegative integer stacks',
                        'legal randomized min/full/short-all-in raises and calls/folds',
                        'suit-permuted hand replay gives identical action states and awards',
                        'independent five-card-combination showdown ranks',
                        'independent per-chip side-pot bucket settlement',
                        'public observation contains hero-only private cards',
                        'unique card deal and valid actor/to-call'],
                wall_seconds=time.perf_counter()-started)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hands', type=int, default=2000)
    p.add_argument('--seed', type=int, default=9077)
    p.add_argument('--out')
    args = p.parse_args()
    result = audit(args.hands, args.seed)
    content = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(content+'\n', encoding='utf-8')
    print(content)
