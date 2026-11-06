"""Independent equity-aware benchmark policies, designed without candidate tuning.

These use shared equity estimates, so their errors are correlated with the bot's
equity model. The policies are separate heuristics, not equilibrium solutions.
Do not select candidate parameters using their held-out challenge results.
"""
from random import Random

from bot.equity import estimate, preflop_percentile
from bot.evaluator import evaluate


OPEN_WIDTH = {
    'tight': (.46, .35, .28, .22, .18, .15, .12, .10),
    'balanced': (.70, .50, .40, .32, .27, .22, .18, .15),
    'loose': (.85, .70, .58, .47, .40, .34, .28, .24),
}


def _forced_chop(board):
    if len(board) != 5:
        return False
    value = evaluate(board)
    category, value = divmod(value, 759375)
    first, value = divmod(value, 50625)
    second = value // 3375
    if category == 8:
        return first == 14
    if category == 7:
        return second == (13 if first == 14 else 14)
    return category == 4 and first == 14 and max(sum(c % 4 == s for c in board) for s in range(4)) <= 2


def _pot_share(obs, equity):
    """Approximate expected share of hero-eligible final call pots by layer.

    Independent identical-opponent equity is a deliberate simplification. Each
    side pot includes only live opponents whose commitments reach that layer.
    """
    hero = obs['hero']
    call = min(obs['to_call'], obs['stacks'][hero])
    paid = list(obs['contributions'])
    paid[hero] += call
    cap = paid[hero]
    levels = sorted({min(c, cap) for c in paid if c > 0})
    previous = 0
    expected = total = 0.0
    for level in levels:
        contributors = [i for i, c in enumerate(paid) if c >= level]
        amount = (level - previous) * len(contributors)
        opponents = sum(obs['in_hand'][i] and i != hero for i in contributors)
        expected += amount * equity ** opponents
        total += amount
        previous = level
    return expected, total


class EquityBaseline:
    """Position-aware opening, pot-odds defense and modest value/bluff betting."""

    def __init__(self, style='balanced', seed=0):
        if style not in OPEN_WIDTH:
            raise ValueError('style must be tight, balanced, or loose')
        self.style = style
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = Random(self.seed)

    @staticmethod
    def _pass(obs, call=False):
        cost = min(obs['to_call'], obs['stacks'][obs['hero']])
        return {'action': 'call' if call and cost else 'fold' if cost else 'check',
                'amount': cost if call else 0}

    def _raise(self, obs, total):
        if not obs['can_raise']:
            return self._pass(obs, True)
        total = min(obs['max_raise_to'], max(obs['min_raise_to'], round(total)))
        current = obs['street_bets'][obs['hero']]
        if total <= current + min(obs['to_call'], obs['stacks'][obs['hero']]):
            return self._pass(obs, True)
        return {'action': 'raise', 'amount': total}

    @staticmethod
    def _position(obs):
        hero, button = obs['hero'], obs['button']
        n = len(obs['stacks'])
        opponents = [i for i, live in enumerate(obs['in_hand']) if live and i != hero]
        order = lambda i: (i - button - 1) % n
        ip = all(order(hero) > order(i) for i in opponents)
        if 'position' in obs:
            return obs['position'], ip
        alive = [i for i in range(n) if obs['stacks'][i] + obs['contributions'][i] > 0]
        if len(alive) == 2:
            preflop = [button] + [i for i in alive if i != button]
        else:
            clockwise = sorted(alive, key=order)
            # If button is empty and SB is dead, explicit adapter position is
            # required for exact preflop order. This fallback assumes live SB.
            preflop = clockwise[2:] + clockwise[:2]
        behind = sum(i in opponents for i in preflop[preflop.index(hero) + 1:])
        return behind, ip

    @staticmethod
    def _range(obs):
        raises = [r for r in obs.get('history', ()) if r['street'] == 0 and r['action'] == 'raise']
        if not raises:
            return .65
        opener = raises[0]['seat']
        fraction = .18 + .16 * (opener == obs['button'])
        fraction *= .48 ** (len(raises) - 1)
        if raises[-1]['amount'] <= obs['big_blind'] * 3:
            fraction += .04
        return max(.04, min(.65, fraction))

    def act(self, obs):
        hero = obs['hero']
        if obs['stacks'][hero] <= 0:
            return self._pass(obs)
        if not obs['board']:
            return self._preflop(obs)
        if _forced_chop(obs['board']):
            return self._pass(obs, True)
        return self._postflop(obs)

    def _preflop(self, obs):
        hero = obs['hero']
        bb = obs['big_blind']
        stack = obs['stacks'][hero]
        cost = min(stack, obs['to_call'])
        rank = preflop_percentile(obs['hole'])
        behind, ip = self._position(obs)
        raises = [r for r in obs.get('history', ()) if r['street'] == 0 and r['action'] == 'raise']
        opponents = [i for i, live in enumerate(obs['in_hand']) if live and i != hero]
        effective = min(stack + obs['street_bets'][hero],
                        max((obs['stacks'][i] + obs['street_bets'][i] for i in opponents), default=0)) / bb
        if not raises and obs['can_raise']:
            width = OPEN_WIDTH[self.style][min(7, behind)]
            limpers = sum(r['street'] == 0 and r['action'] == 'call' for r in obs.get('history', ()))
            width /= 1 + .18 * limpers
            if effective <= 8:
                width = min(width, .15 + .50 / (behind + 1) + .02 * (8 - effective))
                if rank < width:
                    return self._raise(obs, obs['max_raise_to'])
            elif rank < width:
                return self._raise(obs, bb * (2.4 + limpers))
        fraction = self._range(obs)
        eq = estimate(obs['hole'], [], fraction)
        if raises and obs['can_raise'] and rank < (.03 if self.style == 'tight' else .05) and eq > .57:
            size = raises[-1]['amount'] * (2.8 if ip else 3.4)
            if effective < 18:
                size = obs['max_raise_to']
            return self._raise(obs, size)
        if not cost:
            return self._pass(obs)
        # Before opponents have acted, require additional equity to compensate
        # for possible squeezes and inability to realize all showdown equity.
        all_in = cost == stack or all(obs['stacks'][i] == 0 for i in opponents)
        realization = 1.0 if all_in else .90 if ip else .78
        if not raises:
            eq = estimate(obs['hole'], [], 1.0)
        ev, total = _pot_share(obs, eq)
        margin = {'tight': .07, 'balanced': .035, 'loose': .005}[self.style]
        if all_in:
            margin = 0
        return self._pass(obs, ev * realization > cost + margin * total)

    def _postflop_inputs(self, obs):
        hero = obs['hero']
        _, ip = self._position(obs)
        cost = min(obs['to_call'], obs['stacks'][hero])
        equity = estimate(obs['hole'], obs['board'], self._range(obs))
        actions = [r for r in obs.get('history', ()) if r['street'] == obs['street'] and r['action'] == 'raise']
        if cost:
            equity -= (0.08, 0.12, 0.17)[obs['street'] - 1]
            equity -= .04 * min(2, cost / max(1, obs['pot'] - cost))
            equity -= .07 * max(0, len(actions) - 1)
        equity = max(0, equity)
        expected, total = _pot_share(obs, equity)
        opponents = sum(obs['in_hand']) - 1
        # Board/hole flush draws only; no inferred opponent hand information.
        draw = len(obs['board']) < 5 and any(
            sum(c % 4 == suit for c in obs['hole'] + obs['board']) == 4
            and any(c % 4 == suit for c in obs['hole']) for suit in range(4))
        return equity, expected, total, opponents, ip, draw

    def _postflop(self, obs):
        hero = obs['hero']
        stack = obs['stacks'][hero]
        cost = min(obs['to_call'], stack)
        equity, expected, total, opponents, ip, draw = self._postflop_inputs(obs)
        margin = {'tight': .055, 'balanced': .02, 'loose': -.01}[self.style]
        if cost == stack or len(obs['board']) == 5:
            margin = 0
        call = expected >= cost + margin * total
        value = equity ** opponents > (.82 if cost else .66)
        chance = {'tight': .07, 'balanced': .16, 'loose': .27}[self.style]
        bluff = (opponents == 1 and (draw or equity < .34) and self.rng.random() < chance * (1 if ip else .4))
        if cost:
            bluff = draw and opponents == 1 and cost < stack * .10 and self.rng.random() < chance * .5
        if obs['can_raise'] and (value or bluff):
            fraction = .75 if value else .55
            target = obs['street_bets'][hero] + cost + fraction * (obs['pot'] + cost)
            if value and target > obs['max_raise_to'] * .65:
                target = obs['max_raise_to']
            return self._raise(obs, target)
        return self._pass(obs, call)


class PolarizedBettor(EquityBaseline):
    """Configurable large value bets and weak-hand bluffs; chip-EV heuristic.

    bluff_frequency is the chance to bet an eligible weak hand, not the share
    of all bets that are bluffs. No claim of balanced/GTO frequencies is made.
    """

    def __init__(self, seed=0, bluff_frequency=.22, bet_fraction=1.25):
        if not 0 <= bluff_frequency <= 1 or not 0 < bet_fraction <= 4:
            raise ValueError('invalid polarized-bettor parameters')
        self.bluff_frequency = bluff_frequency
        self.bet_fraction = bet_fraction
        super().__init__('balanced', seed)

    def _postflop(self, obs):
        hero = obs['hero']
        cost = min(obs['to_call'], obs['stacks'][hero])
        equity, expected, total, opponents, ip, draw = self._postflop_inputs(obs)
        value = equity ** opponents > (.91 if cost else .76)
        weak = equity < .30 or draw
        chance = self.bluff_frequency * (1 if ip else .65)
        if cost:
            chance *= .18
        bluff = opponents == 1 and weak and self.rng.random() < chance
        if obs['can_raise'] and (value or bluff):
            return self._raise(obs, obs['street_bets'][hero] + cost + self.bet_fraction * (obs['pot'] + cost))
        return self._pass(obs, expected >= cost + .02 * total)
