"""Independent synthetic opponents for local experiments; not contest bots.

These deliberately distinct hand-crafted policies probe obvious exploitability.
They are not equilibrium benchmarks, and beating them proves no contest edge.
"""
import random

BASELINES = ('station', 'nit', 'loose', 'maniac', 'random', 'value', 'mc', 'reference')

# Freeze this bot's own initial policy parameters before final selection.
# This does not preserve historical source/model weights: the reference uses
# the current Agent implementation and selected model with these fixed knobs.
# Keep every original key explicit; do not derive these from mutable DEFAULTS.
BASE_REFERENCE_PARAMS = {
    'open_scale': 1.0, 'call_margin': .025, 'value_threshold': .69,
    'bluff_scale': 1.0, 'risk_scale': 1.0, 'cbet': .54,
    'postflop_range_weight': .60, 'shove_max_bb': 11.0,
    'conditional_weight': .75, 'bluff_prior': .18, 'bet_penalty_scale': 1.0,
}


def preflop_strength(hole):
    hi, lo = sorted((c // 4 + 2 for c in hole), reverse=True)
    if hi == lo:
        return .49 + .033 * hi
    return min(.96, .12 + .026 * hi + .016 * lo
               + .045 * (hole[0] % 4 == hole[1] % 4)
               + .018 * max(0, 4-(hi-lo)))


def postflop_strength(hole, board):
    ranks = [c // 4 + 2 for c in hole + board]
    counts = {r: ranks.count(r) for r in set(ranks)}
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    suits = [sum(c % 4 == s for c in hole+board) for s in range(4)]
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    runs = [len(unique.intersection(range(start, start+5))) for start in range(1, 11)]
    if max(suits) >= 5 or max(runs) == 5:
        strength = .93
    elif groups[0][0] == 4 or (groups[0][0] == 3 and len(groups) > 1 and groups[1][0] >= 2):
        strength = .98
    elif groups[0][0] == 3:
        strength = .83 if any(c//4+2 == groups[0][1] for c in hole) else .48
    elif len(groups) > 1 and groups[0][0] == groups[1][0] == 2:
        strength = .77 if any(counts[c//4+2] == 2 for c in hole) else .32
    elif groups[0][0] == 2:
        pair_rank = groups[0][1]
        private = any(c//4+2 == pair_rank for c in hole)
        strength = (.66 if pair_rank >= max(c//4+2 for c in board) else .47) if private else .27
    else:
        strength = .23 + .011 * max(c//4+2 for c in hole)
    if len(board) < 5:
        flush_draw = any(suits[s] == 4 and any(c % 4 == s for c in hole) for s in range(4))
        strength += .13 * flush_draw + .07 * (max(runs) == 4)
    return min(.995, strength)


class Baseline:
    def __init__(self, style, seed=0):
        if style not in BASELINES:
            raise ValueError(style)
        self.style = style
        self.rng = random.Random(seed)
        self.errors = 0

    @staticmethod
    def _pass(obs):
        return {'action': 'call' if obs['to_call'] else 'check'}

    @staticmethod
    def _raise(obs, amount):
        if not obs['can_raise']:
            return Baseline._pass(obs)
        return {'action': 'raise', 'amount': int(min(obs['max_raise_to'], max(obs['min_raise_to'], amount)))}

    def act(self, obs):
        style, rng = self.style, self.rng
        call, pot, bb = obs['to_call'], obs['pot'], obs['bb']
        hero = obs['hero']
        current = obs['committed'][hero]
        stack = obs['stacks'][hero]
        strength = (preflop_strength(obs['hole']) if not obs['board'] else
                    postflop_strength(obs['hole'], obs['board']))
        if style == 'random':
            x = rng.random()
            if x < .19 and obs['can_raise']:
                return self._raise(obs, obs['max_raise_to'] if x < .035 else current+call+max(bb, pot)*rng.uniform(.4, 1.5))
            return {'action': 'fold'} if call and x < .5 else self._pass(obs)
        if style == 'station':
            if strength > .88 and rng.random() < .45:
                return self._raise(obs, current+call+max(bb, pot*.5))
            if call > .7 * stack and strength < .44:
                return {'action': 'fold'}
            return self._pass(obs)
        if style == 'maniac':
            if obs['can_raise'] and rng.random() < .57:
                return self._raise(obs, obs['max_raise_to'] if stack < 12*bb or rng.random() < .09
                                   else current+call+max(3*bb, pot*rng.uniform(.7, 1.5)))
            return {'action': 'fold'} if call > .35*stack and strength < .40 and rng.random() < .6 else self._pass(obs)
        threshold = {'nit': .73, 'value': .63, 'loose': .52}[style]
        bet_threshold = threshold + (.06 if obs['board'] else .01)
        if stack <= 9*bb and strength >= threshold-.025:
            return self._raise(obs, obs['max_raise_to'])
        if obs['can_raise'] and strength > bet_threshold:
            size = current+call+max(bb, pot * (.65 if obs['board'] else 1.4))
            return self._raise(obs, size)
        if not call:
            if style == 'loose' and obs['board'] and rng.random() < .13:
                return self._raise(obs, current+max(bb, pot*.55))
            return {'action': 'check'}
        odds = call/(pot+call)
        if not obs['board']:
            cutoff = threshold-.07 + min(.18, call/max(bb, stack)*.35)
            if call <= bb and style == 'loose':
                cutoff -= .06
        else:
            cutoff = max(threshold-.17, .34 + .65*odds)
            cutoff += .10 * (call > .55*stack)
        return self._pass(obs) if strength >= cutoff else {'action': 'fold'}


def make_baseline(name, seed=0, model_path=None):
    if name == 'reference':
        from bot import Agent
        agent = Agent(seed=seed, params=BASE_REFERENCE_PARAMS, model_path=model_path)
        # Explicit replacement also excludes keys added to future DEFAULTS.
        agent.params = dict(BASE_REFERENCE_PARAMS)
        return agent
    return MonteCarloAgent(seed=seed) if name == 'mc' else Baseline(name, seed)


class MonteCarloAgent:
    """Stronger benchmark-only opponent; deliberately NOT a 1ms submission.

    Samples joint private cards and board runouts without replacement from the
    public observation. Preflop ranges respond to anonymous action history;
    postflop aggressive hands receive a soft made-hand/draw likelihood weight.
    Call EV sums only side pots the hero can contest, with folded dead money.
    This is an independent heuristic equity opponent, not an equilibrium solver.
    """
    def __init__(self, seed=0, samples=96):
        self.rng = random.Random(seed)
        self.samples = samples
        self.errors = 0

    def _ranges(self, obs, villains):
        ranges = {i: 1.0 for i in villains}
        post_aggression = {i: 0 for i in villains}
        raises = 0
        for entry in obs['history']:
            seat = entry['seat']
            if entry['street'] == 0:
                if entry['action'] == 'raise':
                    raises += 1
                    if seat in ranges:
                        ranges[seat] = min(ranges[seat], (.22, .10, .045)[min(raises-1, 2)])
                elif entry['action'] == 'call' and seat in ranges:
                    ranges[seat] = min(ranges[seat], .40 if raises else .75)
            elif seat in ranges and entry['action'] == 'raise':
                post_aggression[seat] += 1
        return ranges, post_aggression

    def estimate(self, obs):
        from equity import preflop_percentile
        from evaluator import evaluate
        hero = obs['hero']
        villains = [i for i in range(len(obs['stacks'])) if i != hero and not obs['folded'][i]]
        ranges, aggression = self._ranges(obs, villains)
        known = set(obs['hole'] + obs['board'])
        unknown = [c for c in range(52) if c not in known]
        contributions = list(obs['contributions'])
        contributions[hero] += obs['to_call']
        # Build independently of ranks, so every rollout uses identical pots.
        layers = []
        previous = 0
        for level in sorted(set(contributions) - {0}):
            paid = [i for i, value in enumerate(contributions) if value >= level]
            amount = (level-previous) * len(paid)
            previous = level
            if contributions[hero] >= level:
                eligible = [i for i in paid if not obs['folded'][i]]
                layers.append((amount, eligible))
        possible_award = sum(amount for amount, _ in layers)
        win_share = expected_return = 0.
        for _ in range(self.samples):
            remaining = list(unknown)
            holes = {}
            for seat in sorted(villains, key=lambda i: ranges[i]):
                # Bounded soft rejection retains nonzero support for any hand.
                for attempt in range(40):
                    hand = self.rng.sample(remaining, 2)
                    pre_weight = 1. if preflop_percentile(hand) <= ranges[seat] else .04
                    post_weight = 1.
                    if aggression[seat] and obs['board']:
                        s = postflop_strength(hand, obs['board'])
                        post_weight = min(1., .10 + max(0., s-.3) * (1.7 if aggression[seat] == 1 else 1.4))
                    if self.rng.random() < pre_weight*post_weight or attempt == 39:
                        break
                holes[seat] = hand
                remaining.remove(hand[0])
                remaining.remove(hand[1])
            board = obs['board'] + self.rng.sample(remaining, 5-len(obs['board']))
            ranks = {hero: evaluate(obs['hole']+board)}
            ranks.update((seat, evaluate(hand+board)) for seat, hand in holes.items())
            best = max(ranks.values())
            if ranks[hero] == best:
                win_share += 1./sum(rank == best for rank in ranks.values())
            for amount, eligible in layers:
                best = max(ranks[seat] for seat in eligible)
                if ranks[hero] == best:
                    expected_return += amount/sum(ranks[seat] == best for seat in eligible)
        return win_share/self.samples, expected_return/self.samples, possible_award

    def act(self, obs):
        from equity import preflop_percentile
        hero = obs['hero']
        call, pot, bb = obs['to_call'], obs['pot'], obs['bb']
        stack, current = obs['stacks'][hero], obs['committed'][hero]
        active = sum(not folded for folded in obs['folded'])
        # Range-led opening avoids treating all yet-to-act seats as forced callers.
        if not obs['board'] and not any(e['action'] == 'raise' for e in obs['history']):
            left = sum(not obs['folded'][i] and (i-hero) % len(obs['stacks']) < (obs['button']-hero) % len(obs['stacks'])
                       for i in range(len(obs['stacks'])) if i != hero)
            opening = min(.65, .18 + .035*(8-active) + .025*max(0, 5-left))
            if preflop_percentile(obs['hole']) <= opening and obs['can_raise']:
                target = obs['max_raise_to'] if stack <= 11*bb else current+call+2.2*bb
                return Baseline._raise(obs, target)
            if call > 0 and current < bb and preflop_percentile(obs['hole']) > opening+.1:
                return {'action': 'fold'}
        equity, award, possible = self.estimate(obs)
        realization_margin = .025 if len(obs['board']) < 5 else .005
        call_ev = award-call
        value_threshold = max(.34, 1./max(2, active)+.15)
        if obs['can_raise'] and equity > value_threshold:
            size = current+call+max(bb, .7*max(pot, 3*bb))
            if stack <= 8*bb or current+call+pot >= obs['max_raise_to']*.7:
                size = obs['max_raise_to']
            return Baseline._raise(obs, size)
        if call:
            if call_ev >= realization_margin*max(1, possible):
                return {'action': 'call'}
            return {'action': 'fold'}
        if obs['can_raise'] and obs['board'] and active <= 3 and self.rng.random() < .08:
            return Baseline._raise(obs, current+max(bb, pot*.55))
        return {'action': 'check'}
