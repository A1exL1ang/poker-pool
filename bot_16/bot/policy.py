"""Position, range and stack aware NLHE policy with offline equity inference.

This is an independently developed empirical policy, not a Nash equilibrium.
Only public actions in the current table affect the opponent estimates.
"""
import math
import random

from .equity import percentile, preflop, postflop
from .evaluator import evaluate, STRAIGHT
from .features import make_features
from .icm import call_threshold


DEFAULTS = {
    'open_scale': 1.0,
    'call_margin': 0.05,
    'value_threshold': 0.57,
    'bluff_scale': 1.0,
    'pressure_scale': 1.3,
    'raise_threshold': 0.75,
    'bet_fraction': 0.62,
    'multiway_discount': 0.0,
    'loose_raises': 1.0,
}


def normalize(action, obs):
    """Legal action guard; raise amount is TOTAL street contribution."""
    call = max(0, obs.get('to_call', 0))
    kind = action.get('action') if isinstance(action, dict) else None
    passive = {'action': 'check' if call == 0 else 'fold'}
    if kind == 'fold':
        return passive
    if kind in ('check', 'call'):
        return {'action': 'call' if call else 'check'} if kind == 'call' or not call else passive
    if kind != 'raise' or not obs.get('can_raise', False):
        return passive
    hero = obs['players'][obs['hero']]
    maximum = min(int(obs['max_raise_to']), int(hero['street_bet'] + hero['stack']))
    minimum = int(obs['min_raise_to'])
    current = int(hero['street_bet'] + call)
    try:
        target = int(action['amount'])
    except (ValueError, TypeError, KeyError, OverflowError):
        return passive
    if maximum <= current:
        return {'action': 'call' if call else 'check'}
    target = min(maximum, max(minimum, target))
    return {'action': 'raise', 'amount': target}


class Agent:
    def __init__(self, seed=16, params=None, strict=False, diagnostics=False):
        self.seed = seed
        self.params = dict(DEFAULTS)
        if params:
            unknown = set(params) - set(DEFAULTS)
            if unknown:
                raise ValueError('Unknown policy parameters: ' + str(unknown))
            self.params.update(params)
        self.strict = strict
        self.diagnostics = diagnostics
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.hand_id = None
        self.processed = 0
        self.seen = set()
        self.vpipped = set()
        self.preflop_raises = []
        self.limp_count = 0
        self.roles = [[0, 0, False] for _ in range(8)]
        self.post_bets = []
        self.stats = [[0, 0, 0, 0, 0, 0] for _ in range(8)]
        self.errors = 0
        self.fallbacks = 0
        self.last_error = None
        self.last_info = None

    def act(self, observation):
        try:
            return normalize(self._act(observation), observation)
        except Exception as exc:
            self.errors += 1
            self.fallbacks += 1
            self.last_error = type(exc).__name__ + ': ' + str(exc)
            if self.strict:
                raise
            return {'action': 'check' if observation.get('to_call', 0) == 0 else 'fold'}

    def _observe(self, obs):
        history = obs.get('history', ())
        if obs.get('hand_id') != self.hand_id:
            self.hand_id = obs.get('hand_id')
            self.processed = 0
            self.seen.clear()
            self.vpipped.clear()
            self.preflop_raises.clear()
            self.limp_count = 0
            self.roles = [[0, 0, False] for _ in range(8)]
            self.post_bets.clear()
        if len(history) < self.processed:
            # A local adapter may deliver shorter histories; avoid double count.
            self.processed = len(history)
        for action in history[self.processed:]:
            seat, street, kind = action['seat'], action['street'], action['action']
            stat = self.stats[seat]
            if street == 0:
                if seat not in self.seen:
                    stat[0] += 1
                    self.seen.add(seat)
                if kind in ('call', 'raise') and seat not in self.vpipped:
                    stat[1] += 1
                    self.vpipped.add(seat)
                if kind == 'raise':
                    stat[2] += 1
                    self.preflop_raises.append(action)
                    self.roles[seat][0] = len(self.preflop_raises)
                    self.roles[seat][1] = action['to']
                elif kind == 'call':
                    self.limp_count += 1
                    self.roles[seat][2] = True
            else:
                stat[3] += 1
                stat[4] += kind == 'raise'
                stat[5] += kind == 'fold'
                if kind == 'raise':
                    self.post_bets.append(action)
        self.processed = len(history)

    def _vpip(self, seat):
        stat = self.stats[seat]
        return (stat[1] + 6.0) / (stat[0] + 20.0)

    def _aggression(self, seat):
        stat = self.stats[seat]
        return (stat[4] + 5.0) / (stat[3] + 20.0)

    def _position(self, obs):
        players, hero, button = obs['players'], obs['hero'], obs['button']
        ids = [i for i, p in enumerate(players) if p['stack'] + p['invested'] > 0]
        clockwise = sorted(ids, key=lambda i: (i - button) % len(players) or len(players))
        # Clockwise starts left of button and ends at button.
        if len(ids) == 2:
            pre = [button] + [i for i in ids if i != button]
        else:
            pre = clockwise[2:] + clockwise[:2]
        behind = sum(not players[i]['folded'] for i in pre[pre.index(hero) + 1:])
        post = [i for i in clockwise if not players[i]['folded'] and players[i]['stack'] > 0]
        in_position = not post or post[-1] == hero
        return behind, in_position

    def _range(self, obs, seat):
        bb = obs['big_blind']
        level, size, called = self.roles[seat]
        vpip = self._vpip(seat)
        stat = self.stats[seat]
        loose_floor = 0.0
        if stat[0] >= 10 and stat[1] / stat[0] > 0.65 and stat[2] / stat[0] > 0.60:
            # Strong table-local evidence can overturn a tight sizing prior.
            # Shrink heavily at small samples and never identify an opponent
            # beyond its current anonymous seat.
            loose_floor = self.params['loose_raises'] * min(0.9, (stat[2] + 3) / (stat[0] + 10)) * stat[0] / (stat[0] + 10)
        if level:
            if level >= 3:
                return max(loose_floor, 0.025, min(0.13, vpip * 0.18))
            if level == 2:
                return max(loose_floor, 0.055, min(0.23, vpip * 0.34))
            p = obs['players'][seat]
            total_bb = (p['stack'] + p['invested']) / bb
            if p.get('all_in') and total_bb <= 12:
                return max(loose_floor, min(0.65, max(0.16, vpip * (1.15 + max(0, 8 - total_bb) / 10))))
            factor = 0.8 if size <= 4 * bb else max(0.22, 3 * bb / size)
            return max(loose_floor, 0.055, min(0.6, vpip * factor))
        return min(1.0, max(0.18, vpip * 1.6)) if called else 1.0

    def _pressure(self, obs, opponents):
        street = obs['street']
        pressure = 0.0
        current_raises = 0
        for a in self.post_bets:
            if a['seat'] not in opponents:
                continue
            fraction = a['paid'] / max(obs['big_blind'], a['pot_before'])
            aggr = self._aggression(a['seat'])
            tight = max(0.55, min(1.45, 0.32 / aggr))
            if a['street'] == street:
                current_raises += 1
                pressure += (0.55 + 0.55 * min(1.5, fraction)) * tight
            else:
                pressure += 0.20 * tight
        # A common pressure parameter applies to every modeled opponent;
        # dilute one bettor's evidence when other opponents remain unfiltered.
        pressure /= max(1.0, len(opponents) ** 0.65)
        return min(2.0, pressure * self.params['pressure_scale']), current_raises

    def _act(self, obs):
        hero = obs['hero']
        players = obs['players']
        p = players[hero]
        hole, board = obs['hole'], obs['board']
        if len(hole) != 2 or len(board) not in (0, 3, 4, 5) or len(set(hole + board)) != len(hole) + len(board):
            raise ValueError('Invalid hole/board cards')
        if any(type(c) is not int or not 0 <= c < 52 for c in hole + board):
            raise ValueError('Cards must be distinct integers 0..51')
        if len(board) != (0, 3, 4, 5)[obs['street']] or obs['big_blind'] <= 0:
            raise ValueError('Invalid street or big blind')
        if not 2 <= len(players) <= 8 or not 0 <= hero < len(players):
            raise ValueError('Expected 2..8 seats')
        self._observe(obs)
        opponents = [i for i, v in enumerate(players) if i != hero and not v['folded'] and v['stack'] + v['invested'] > 0]
        if not opponents or p['stack'] <= 0:
            return {'action': 'check'}
        behind, ip = self._position(obs)
        if not board:
            return self._preflop(obs, opponents, behind, ip)
        return self._postflop(obs, opponents, ip)

    def _raise(self, obs, target):
        return {'action': 'raise', 'amount': int(target)}

    def _preflop(self, obs, opponents, behind, ip):
        p = obs['players'][obs['hero']]
        bb, stack, call = obs['big_blind'], p['stack'], min(p['stack'], obs['to_call'])
        pot, hole = obs['pot'], obs['hole']
        rank = percentile(hole)
        raises = self.preflop_raises
        limpers = self.limp_count
        effective = min(stack, max(obs['players'][i]['stack'] + max(0, obs['players'][i]['street_bet'] - p['street_bet']) for i in opponents))
        depth = effective / bb
        can_raise = obs.get('can_raise', False)
        free = {'action': 'check' if not call else 'fold'}
        maxbet = obs['max_raise_to']
        if not raises:
            opening = (0.8, 0.5, 0.43, 0.31, 0.25, 0.21, 0.18, 0.16)[min(behind, 7)]
            opening *= self.params['open_scale']
            if limpers:
                opening = min(opening, 0.33 / (1 + 0.12 * (limpers - 1)))
            # With short stacks, compare jam to folding using explicit call/fold
            # mixture. This is a simplified shove model, not a Nash chart.
            if depth <= 12 and can_raise and not any(obs['players'][i]['all_in'] for i in opponents):
                n = max(1, behind if not limpers else len(opponents))
                calling_range = min(0.40, 0.11 + 0.015 * max(0, 12 - depth) + 0.04 * (n <= 2))
                fold_prob = (1 - calling_range) ** n
                eq = preflop(hole, calling_range, 1)
                matched = min(effective, stack)
                called_ev = eq * (pot + matched * 2 - call) - matched
                ev = fold_prob * pot + (1 - fold_prob) * called_ev
                # Multi-caller and ICM uncertainty require a small positive edge.
                if rank < opening * 1.25 and ev > bb * (0.10 + 0.05 * max(0, n - 2)):
                    return self._raise(obs, maxbet)
            if can_raise and rank < opening and (depth > 12 or rank < 0.08):
                sizing = (2.2 if depth < 35 else 2.5) + 0.9 * limpers
                if behind == 1:
                    sizing += 0.4
                target = bb * sizing
                if target >= p['street_bet'] + effective * 0.36 and depth < 18:
                    target = maxbet
                return self._raise(obs, target)
            # Complete cheaply in the small blind / overlimp good implied-odds
            # hands; never pay a full blind merely because a weak hand is suited.
            r1, r2 = hole[0] // 4 + 2, hole[1] // 4 + 2
            suited = hole[0] % 4 == hole[1] % 4
            speculative = r1 == r2 or (suited and abs(r1 - r2) <= 2 and min(r1, r2) >= 5)
            if call and (call <= bb / 2 and rank < 0.70 or limpers and speculative and depth >= 18 and call <= bb):
                return {'action': 'call'}
            return free
        aggressor = raises[-1]['seat']
        fraction = self._range(obs, aggressor)
        # Players who have already matched a raise are real competitors;
        # seats yet to act are accounted for by a positional realization penalty.
        matched = [i for i in opponents if obs['players'][i]['street_bet'] > bb or obs['players'][i]['all_in']]
        n = max(1, len(matched))
        eq = preflop(hole, fraction, n)
        cap = p['invested'] + call
        eligible = sum(min(v['invested'], cap) for v in obs['players'])
        odds = call / max(1, eligible + call)
        allin = call == stack or all(obs['players'][i]['stack'] == 0 for i in opponents)
        if allin and len(opponents) == 1:
            icm = call_threshold(obs, opponents[0])
            if icm is not None:
                odds = max(odds, icm)
        if not allin:
            realization = (0.95 if ip else 0.87) - 0.025 * min(3, behind)
            # High pairs and suited cards retain more equity after the flop.
            if hole[0] // 4 == hole[1] // 4 or hole[0] % 4 == hole[1] % 4:
                realization += 0.04
            eq *= min(1.0, realization)
        margin = 0.006 if allin else self.params['call_margin']
        if self.diagnostics:
            self.last_info = {'equity': eq, 'odds': odds, 'range': fraction, 'opponents': n}
        if eq < odds + margin:
            return free
        # Value 3/4-bets; occasional suited-ace bluff only against small opens
        # with room to fold, so huge bets never induce light re-raises.
        value = (rank < 0.045 and len(raises) == 1) or rank < 0.014
        short_value = depth < 18 and eq > max(0.53, odds + 0.15) and rank < 0.2
        bluff = len(raises) == 1 and depth > 35 and n == 1 and raises[-1]['to'] <= 3 * bb and max(hole) // 4 == 12 and hole[0] % 4 == hole[1] % 4 and rank < 0.32 and self.rng.random() < 0.10
        if can_raise and (value or short_value or bluff):
            target = raises[-1]['to'] * (3.0 if ip else 3.7) + max(0, n - 1) * raises[-1]['to']
            if short_value or target >= p['street_bet'] + effective * 0.38:
                target = maxbet
            return self._raise(obs, target)
        return {'action': 'call' if call else 'check'}

    def _postflop(self, obs, opponents, ip):
        p, hole, board = obs['players'][obs['hero']], obs['hole'], obs['board']
        street, pot, bb = obs['street'], obs['pot'], obs['big_blind']
        call = min(p['stack'], obs['to_call'])
        n = len(opponents)
        fraction = math.exp(sum(math.log(self._range(obs, i)) for i in opponents) / n)
        pressure, raises = self._pressure(obs, opponents)
        features = make_features(hole, board, n, fraction, pressure)
        eq = postflop(hole, board, fraction, n, pressure=pressure, features=features)
        score = evaluate(hole + board)
        category = score >> 20
        locked_chop = False
        if category == 8 and (score >> 16) & 15 == 14:
            locked_chop = len(board) == 5 and evaluate(board) == score
            eq = 1 / (n + 1) if locked_chop else 1.0
        if len(board) == 5 and not locked_chop:
            bs = evaluate(board)
            bc, top, kicker = bs >> 20, (bs >> 16) & 15, (bs >> 12) & 15
            locked_chop = bc == 7 and kicker == (13 if top == 14 else 14)
            if bc == 4 and top == 14:
                # Broadway with fewer than three board cards in any suit:
                # neither a better straight, flush, nor paired hand is possible.
                locked_chop = max(sum(c % 4 == s for c in board) for s in range(4)) <= 2
        cap = p['invested'] + call
        eligible = sum(min(v['invested'], cap) for v in obs['players'])
        odds = call / max(1, eligible + call)
        if locked_chop:
            # Everyone's best hand is fixed. Calling cannot lose matched chips;
            # generic uncertainty margins would incorrectly fold large wagers.
            return {'action': 'call' if call else 'check'}
        allin = call == p['stack'] or all(obs['players'][i]['stack'] == 0 for i in opponents)
        if allin and n == 1:
            icm = call_threshold(obs, opponents[0])
            if icm is not None:
                odds = max(odds, icm)
        realized = eq
        if street < 3 and not allin:
            realized *= 0.97 if ip else 0.91
        margin = 0.005 if allin or street == 3 else self.params['call_margin']
        if self.diagnostics:
            self.last_info = {'equity': eq, 'realized': realized, 'odds': odds, 'pressure': pressure, 'range': fraction, 'opponents': n}
        if call:
            if realized < odds + margin:
                return {'action': 'fold'}
            if obs.get('can_raise') and eq > self.params['raise_threshold'] + min(0.09, (n - 1) * 0.025) and self.rng.random() < 0.86:
                target = p['street_bet'] + call + (pot + call) * (0.7 if n == 1 else 0.85)
                if target > p['street_bet'] + p['stack'] * 0.62:
                    target = obs['max_raise_to']
                return self._raise(obs, target)
            return {'action': 'call'}
        if not obs.get('can_raise'):
            return {'action': 'check'}
        # Equity conditional on someone continuing drives value sizing.
        called = postflop(hole, board, fraction, n, pressure=min(2.0, pressure + 0.85), features=features)
        if eq == 1.0:
            called = 1.0
        value_threshold = (self.params['value_threshold'] + (0.025 if n > 1 else 0)) * (2.0 / (n + 1)) ** self.params['multiway_discount']
        value = called > value_threshold
        suits = [0, 0, 0, 0]
        mask = 0
        for c in hole + board:
            suits[c % 4] += 1
            mask |= 1 << (c // 4)
        flush_draw = street < 3 and max(suits) == 4 and any(suits[c % 4] == 4 for c in hole)
        straight_draw = street < 3 and not STRAIGHT[mask] and any(STRAIGHT[mask | (1 << rank)] for rank in range(13) if not mask & (1 << rank))
        draw = flush_draw or straight_draw
        # Small, stochastic, blocker/draw-weighted bluff budget. Stronger
        # multiway ranges and poor position sharply suppress speculation.
        bluff_rate = (0.28 if draw else 0.085) if street < 3 else 0.075
        if not ip:
            bluff_rate *= 0.5
        bluff_rate *= 0.33 ** (n - 1)
        if pressure > 0.6:
            bluff_rate *= 0.45
        if eq > 0.48 and not draw:
            bluff_rate *= 0.35  # preserve showdown value
        if max(hole) // 4 == 12:
            bluff_rate *= 1.2
        bluff = self.rng.random() < bluff_rate * self.params['bluff_scale']
        if value or bluff:
            fraction_bet = self.params['bet_fraction']
            if called > 0.85:
                fraction_bet = 0.85
            elif street == 1 and n == 1:
                fraction_bet = 0.42
            if n > 2:
                fraction_bet = max(fraction_bet, 0.70)
            target = p['street_bet'] + max(bb, pot * fraction_bet)
            if target > p['street_bet'] + p['stack'] * 0.65 and value:
                target = obs['max_raise_to']
            return self._raise(obs, target)
        return {'action': 'check'}
