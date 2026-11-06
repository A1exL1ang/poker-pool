"""bot_17: bounded, offline no-limit Hold'em strategy.

This is a LOCAL interface, documented in CONTRACT.md, not an official adapter.
"""
import math
import random

import equity
from risk import call_threshold


DEFAULTS = dict(open_scale=1.0, value_margin=0.04, call_margin=0.035,
                bluff_scale=1.0, samples=96, adapt=True, icm=True)


def _pre_score(a, b):
    r, s = sorted((a // 4 + 2, b // 4 + 2), reverse=True)
    if r == s:
        return 0.50 + r * 0.025
    score = 0.25 + r * 0.020 + s * 0.009
    score += 0.035 if a % 4 == b % 4 else 0
    gap = r - s
    score += 0.020 if gap == 1 else (0.010 if gap == 2 else -0.008 * min(4, gap - 2))
    if r == 14 and s <= 5:
        score += 0.018
    return score


# Combo-weighted percentile, built once at import, never during act.
_classes = []
for _r in range(13):
    for _s in range(_r + 1):
        for _suited in ((False,) if _r == _s else (False, True)):
            _a, _b = _r * 4, _s * 4 + (0 if _suited else 1)
            _classes.append((_pre_score(_a, _b), (_r, _s, _suited), 6 if _r == _s else (4 if _suited else 12)))
_classes.sort(reverse=True)
PERCENTILE = {}
_count = 0
for _score, _key, _weight in _classes:
    PERCENTILE[_key] = (_count + _weight / 2) / 1326
    _count += _weight


def percentile(hole):
    a, b = hole
    r, s = sorted((a // 4, b // 4), reverse=True)
    return PERCENTILE[r, s, a % 4 == b % 4 if r != s else False]


def _texture(hole, board):
    """Draws and blockers, separating hole contribution from board texture."""
    ranks = [c // 4 + 2 for c in hole + board]
    br = [c // 4 + 2 for c in board]
    suits = [sum(c % 4 == s for c in hole + board) for s in range(4)]
    bs = [sum(c % 4 == s for c in board) for s in range(4)]
    fd = any(suits[s] == 4 and any(c % 4 == s for c in hole) for s in range(4)) and len(board) < 5
    rd = set(ranks)
    if 14 in rd:
        rd.add(1)
    straight_outs = set()
    if len(board) < 5:
        for start in range(1, 11):
            seq = set(range(start, start + 5))
            missing = seq - rd
            if len(missing) == 1 and any(c // 4 + 2 in seq or (c // 4 == 12 and 1 in seq) for c in hole):
                straight_outs |= missing
    draw = min(0.42, (0.18 if fd else 0) + min(2, len(straight_outs)) * 0.055)
    wet = max(bs, default=0) >= 2 or len(set(br)) >= 3 and max(br) - min(br) <= 5
    nut_blocker = any(c // 4 == 12 and bs[c % 4] >= 2 for c in hole)
    return draw, wet, nut_blocker


class Agent:
    def __init__(self, seed=17, config=None):
        self.config = DEFAULTS.copy()
        if config:
            self.config.update(config)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        elif not hasattr(self, 'seed'):
            self.seed = 17
        self.rng = random.Random(self.seed)
        self.hand_id = None
        self.table_id = None
        self.seen = 0
        self.stats = {}
        self.pf_seen = set()
        self.last_aggressor = None
        self.decisions = 0
        self.fallbacks = 0

    def _observe(self, o):
        tid = o.get('table_id')
        if tid is not None and self.table_id is not None and tid != self.table_id:
            self.reset(self.rng.getrandbits(32))
        self.table_id = tid
        if o['hand_id'] != self.hand_id:
            self.hand_id = o['hand_id']
            self.seen = 0
            self.pf_seen.clear()
            self.last_aggressor = None
        hist = o.get('history', ())
        # History is cumulative within a hand under the local contract.
        for event in hist[self.seen:]:
            seat, act = event['seat'], event['action']
            if seat == o['hero'] or act == 'blind':
                continue
            st = self.stats.setdefault(seat, [0, 0, 0, 0, 0, 0])
            if event['street'] == 0 and seat not in self.pf_seen:
                st[0] += 1
                st[1] += act in ('call', 'raise')
                st[2] += act == 'raise'
                self.pf_seen.add(seat)
            if event['street'] > 0:
                st[3] += 1
                st[4] += act == 'raise'
                st[5] += act == 'fold'
        self.seen = len(hist)

    def _style(self, seat):
        if not self.config['adapt']:
            return .30, .18, .28
        s = self.stats.get(seat, (0, 0, 0, 0, 0, 0))
        # Substantial prior avoids chasing tiny, selectively observed samples.
        return ((s[1] + 12) / (s[0] + 40),
                (s[2] + 7.2) / (s[0] + 40),
                (s[4] + 11.2) / (s[3] + 40))

    def act(self, observation):
        self.decisions += 1
        if not isinstance(observation, dict):
            self.fallbacks += 1
            return {'action': 'fold'}
        try:
            o = observation
            legal = o['legal_actions']
            if 'check' not in legal and 'call' not in legal and 'raise' not in legal:
                return {'action': 'fold'}
            hole, board = o['hole'], o['board']
            if len(hole) != 2 or len(board) not in (0, 3, 4, 5) or len(set(hole + board)) != len(hole + board) or any(type(c) is not int or c < 0 or c > 51 for c in hole + board):
                raise ValueError('Invalid cards')
            players = o['players']
            hero = next(p for p in players if p['seat'] == o['hero'])
            if hero['stack'] < 0 or o['pot'] < 0 or o['to_call'] < 0:
                raise ValueError('Invalid chips')
            self._observe(o)
            live = [p for p in players if not p['folded'] and p['seat'] != o['hero'] and p['stack'] + p['committed'] > 0]
            if not live:
                return self._pass(o)
            if o['street'] == 0:
                action = self._preflop(o, hero, live)
            else:
                action = self._postflop(o, hero, live)
            return self._legalize(o, action)
        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError, StopIteration):
            self.fallbacks += 1
            return {'action': 'check' if 'check' in observation.get('legal_actions', ()) else 'fold'}

    @staticmethod
    def _pass(o):
        return {'action': 'check' if 'check' in o['legal_actions'] else 'fold'}

    @staticmethod
    def _legalize(o, action):
        name = action['action']
        if name == 'raise' and 'raise' in o['legal_actions'] and o['can_raise']:
            maximum = int(o['max_raise_to'])
            minimum = int(o['min_raise_to'])
            amount = min(maximum, max(minimum, int(action['amount'])))
            if amount > max(p['bet'] for p in o['players']):
                return {'action': 'raise', 'amount': amount}
        elif name in o['legal_actions']:
            return {'action': name}
        return Agent._pass(o)

    def _raise(self, o, amount):
        if o['can_raise'] and 'raise' in o['legal_actions']:
            return {'action': 'raise', 'amount': int(amount)}
        return {'action': 'call'} if o['to_call'] else self._pass(o)

    @staticmethod
    def _eligible(o, hero):
        ceiling = hero['committed'] + o['to_call']
        return sum(min(p['committed'], ceiling) for p in o['players'])

    @staticmethod
    def _pot_layers(o, hero, live):
        """Eligible final pots after a call; masks index live opponents.

        Unacted funded opponents remain possible contestants, conservatively;
        an all-in opponent can win only layers its contribution reaches.
        """
        cap = hero['committed'] + o['to_call']
        if not any(p['all_in'] and p['committed'] < cap for p in live):
            return None
        contributions = [min(cap, p['committed'] + (o['to_call'] if p['seat'] == hero['seat'] else 0)) for p in o['players']]
        pots = []
        previous = 0
        for level in sorted(set(contributions)):
            if level <= 0:
                continue
            amount = (level - previous) * sum(c >= level for c in contributions)
            mask = sum(1 << i for i,p in enumerate(live) if not p['all_in'] or p['committed'] >= level)
            if pots and pots[-1][1] == mask:
                pots[-1] = (pots[-1][0] + amount, mask)
            else:
                pots.append((amount, mask))
            previous = level
        return pots

    def _risk(self, o, hero, live, pot):
        call = o['to_call']
        odds = call / max(1, pot + call)
        payouts = o.get('payouts')
        if self.config['icm'] and payouts and len(live) == 1 and call >= hero['stack'] * .40:
            if all(payouts[i] >= payouts[i + 1] >= 0 for i in range(len(payouts) - 1)):
                odds = max(odds, call_threshold(o['players'], o['hero'], live[0]['seat'], call, pot, payouts))
        return odds

    def _preflop(self, o, hero, live):
        bb = max(1, o['big_blind'])
        depth = (hero['stack'] + hero['bet']) / bb
        q = percentile(o['hole'])
        a, b = o['hole']
        pair, suited = a // 4 == b // 4, a % 4 == b % 4
        high, low = sorted((a // 4 + 2, b // 4 + 2), reverse=True)
        players = [p for p in o['players'] if p['stack'] + p['committed'] > 0]
        seats = sorted(p['seat'] for p in players)
        button = o['button']
        ring = max(p['seat'] for p in o['players']) + 1
        clockwise = sorted(seats, key=lambda seat: (seat - button - 1) % ring)
        # Position relative to button: SB, BB, then earliest through button.
        if len(players) == 2:
            late = o['hero'] == button
            width = .78 if late else .55
        else:
            blinds = [e for e in o.get('history', ()) if e['street'] == 0 and e['action'] == 'blind']
            bb_seat = blinds[-1]['seat'] if blinds else clockwise[1]
            sb_seat = blinds[-2]['seat'] if len(blinds) >= 2 else (None if blinds else clockwise[0])
            order = sorted(seats, key=lambda seat: (seat - bb_seat - 1) % ring)
            behind = sum(s not in (sb_seat,bb_seat) for s in order[order.index(o['hero']) + 1:])
            if o['hero'] == sb_seat:
                width = .55
            elif o['hero'] == bb_seat:
                width = .42
            else:
                width = min(.44, .15 + .29 * math.exp(-.48 * behind))
            late = behind <= 1 or o['hero'] == sb_seat
        width *= self.config['open_scale']
        hist = o.get('history', ())
        raises = [e for e in hist if e['street'] == 0 and e['action'] == 'raise']
        limpers = [p for p in live if p['bet'] >= bb and any(e['seat'] == p['seat'] and e['action'] == 'call' and e['street'] == 0 for e in hist)]
        if not raises:
            # Unopened/limped pots: pressure late, avoid large weak multiway pots.
            width /= 1 + .25 * len(limpers)
            if depth <= 10:
                push_width = min(.65, width * (1.25 if depth <= 6 else .90))
                if q < push_width:
                    return self._raise(o, o['max_raise_to'])
            elif q < width:
                size = (2.2 + .9 * len(limpers) + (.4 if not late else 0)) * bb
                if depth <= 18 and q < .08:
                    size = o['max_raise_to']
                return self._raise(o, size)
            # Small pairs/suited connectors can realize cheap multiway odds.
            if limpers and o['to_call'] <= bb and depth >= 18 and (pair or suited and high - low <= 2 and high >= 8):
                return {'action': 'call'} if o['to_call'] else self._pass(o)
            return self._pass(o)

        raiser = raises[-1]['seat']
        vpip, pfr, aggr = self._style(raiser)
        raise_to = max(p['bet'] for p in live)
        short = raise_to / bb <= 15 and any(p['seat'] == raiser and p['all_in'] for p in live)
        if len(raises) == 1:
            rwidth = max(.08, min(.55, pfr * 1.2 + (.08 if short else 0)))
        else:
            rwidth = max(.025, min(.22, pfr * (.40 if len(raises) == 2 else .20)))
        if raise_to > 25 * bb:
            rwidth *= .65
        opponents = max(1, sum(p['bet'] >= raise_to or p['all_in'] for p in live))
        tightness = min(1.0, (1 - rwidth) / .95)
        pots = self._pot_layers(o, hero, live)
        if pots:
            total = sum(amount for amount,mask in pots)
            eq = sum(amount * (equity.preflop(o['hole'], mask.bit_count(), tightness) if mask else 1.) for amount,mask in pots) / total
        else:
            eq = equity.preflop(o['hole'], opponents, tightness)
        pot = self._eligible(o, hero)
        odds = self._risk(o, hero, live, pot)
        effective = min(hero['stack'], max(p['stack'] + max(0, p['bet'] - hero['bet']) for p in live))
        allin = o['to_call'] >= .95 * hero['stack'] or all(p['all_in'] for p in live)
        # Value 3/4-bets; occasional suited wheel-A bluffs when deep and in position.
        premium = q < (.045 if len(raises) == 1 else .012)
        if pair and high >= 13:
            premium = True
        if o['can_raise'] and premium and eq > .51:
            size = max(o['min_raise_to'], raise_to * (2.8 if late else 3.4) + len(limpers) * bb)
            if size - hero['bet'] > .38 * hero['stack'] or depth <= 20:
                size = o['max_raise_to']
            return self._raise(o, size)
        if o['can_raise'] and len(raises) == 1 and late and depth > 35 and suited and high == 14 and low <= 5 and self.rng.random() < .09 * self.config['bluff_scale']:
            return self._raise(o, raise_to * 3)
        realization = (0 if allin else (.065 if late else .105))
        realization += .018 * max(0, opponents - 1)
        if pair or suited:
            realization -= .018 if not allin else 0
        if eq > odds + realization + self.config['call_margin']:
            return {'action': 'call'}
        # Set mining requires a deep effective stack and restrained entry cost.
        if pair and len(raises) == 1 and depth > 35 and effective >= o['to_call'] * 18 and o['to_call'] <= 3 * bb:
            return {'action': 'call'}
        return self._pass(o)

    def _postflop(self, o, hero, live):
        call = o['to_call']
        pot = self._eligible(o, hero) if call else o['pot']
        bb = max(1, o['big_blind'])
        hist = o.get('history', ())
        pf_raises = [e for e in hist if e['street'] == 0 and e['action'] == 'raise']
        current = [e for e in hist if e['street'] == o['street'] and e['action'] == 'raise']
        raiser = current[-1]['seat'] if current else None
        ranges = []
        for p in live:
            vpip, pfr, aggr = self._style(p['seat'])
            was_raiser = any(e['seat'] == p['seat'] for e in pf_raises)
            width = min(.75, max(.12, pfr * 1.6 if was_raiser else vpip * 1.5))
            # Big-blind checks and unraised pots retain broad ranges.
            if not pf_raises:
                width = min(1., width + .35)
            tight = min(.95, (1 - width) / .95)
            pressure = 0.
            for e in hist:
                if e['seat'] == p['seat'] and e['street'] > 0 and e['action'] == 'raise':
                    pressure += .16 + .04 * e['street']
            if p['seat'] == raiser:
                pressure += min(.35, call / max(bb, pot) * .30)
            # Observed aggressive opponents have broader betting ranges.
            pressure = max(0., min(.90, pressure * (.28 / max(.12, aggr)) ** .45))
            ranges.append((tight, pressure))
        draw, wet, blocker = _texture(o['hole'], o['board'])
        samples = min(self.config['samples'], 64) if len(live) >= 4 else self.config['samples']
        eq = equity.estimate(o['hole'], o['board'], len(live), ranges=ranges,
                             seed=self.rng.getrandbits(32), samples=samples,
                             pots=self._pot_layers(o, hero, live))
        remaining = min(hero['stack'], max(p['stack'] + max(0, p['bet'] - hero['bet']) for p in live))
        spr = remaining / max(bb, pot)
        river = o['street'] == 3
        if river:
            board_rank = equity.evaluate(o['board'])
            # With an unbeatable shared board, a call cannot lose chips. ICM
            # binary win/loss approximations must not turn a certain split into
            # an apparent risky call.
            if ((board_rank >> 20 == 8 and (board_rank >> 16) & 15 == 14) or
                (board_rank >> 20 == 4 and (board_rank >> 16) & 15 == 14 and
                 max(sum(c % 4 == s for c in o['board']) for s in range(4)) <= 2) or
                (board_rank >> 20 == 7 and ((board_rank >> 12) & 15 == 14 or
                  (board_rank >> 16) & 15 == 14 and (board_rank >> 12) & 15 == 13))):
                return {'action': 'call'} if call else self._pass(o)
        headsup = len(live) == 1
        odds = self._risk(o, hero, live, pot)
        # Future betting makes marginal equity harder to realize out of position.
        active_seats = [p['seat'] for p in o['players'] if not p['folded'] and not p['all_in']]
        ring = max(p['seat'] for p in o['players']) + 1
        ip = o['hero'] == max(active_seats, key=lambda s: (s - o['button'] - 1) % ring) if active_seats else False
        margin = self.config['call_margin']
        if not river and spr > 1:
            margin += .035 if ip else .060
        margin += .015 * max(0, len(live) - 1)
        if call:
            allin = call >= .95 * hero['stack'] or all(p['all_in'] for p in live)
            if allin:
                margin = .015
            if eq < odds + margin:
                return self._pass(o)
            value = (.72 if headsup else .66) + self.config['value_margin']
            if eq >= value and o['can_raise'] and not all(p['all_in'] for p in live):
                size = hero['bet'] + call + (pot + call) * (.80 if wet else .65)
                if size - hero['bet'] >= hero['stack'] * .45 or spr < 1.3:
                    size = o['max_raise_to']
                return self._raise(o, size)
            # Semi-bluff only when there are opponents who can still fold.
            if not river and headsup and draw >= .18 and eq > .40 and spr > 2 and o['can_raise'] and self.rng.random() < .10 * self.config['bluff_scale']:
                return self._raise(o, hero['bet'] + call + .75 * (pot + call))
            return {'action': 'call'}

        # No benefit in betting into opponents who are all-in.
        if not any(p['stack'] > 0 for p in live):
            return self._pass(o)
        value = (.60 if headsup else .52) + self.config['value_margin']
        if river:
            value += .035
        if eq > value:
            fraction = .65 if wet or not headsup else .48
            if river and eq > .88:
                fraction = .9
            size = max(bb, pot * fraction)
            if spr < 1.1 and eq > .70:
                size = o['max_raise_to']
            return self._raise(o, hero['bet'] + size)
        pfa = pf_raises[-1]['seat'] == o['hero'] if pf_raises else False
        # Small dry-board continuation bets and blocker/draw bluffs, mixed.
        bluff = 0.
        if headsup:
            if not river and (draw > .10 or pfa and not wet):
                bluff = .30 if ip else .16
            elif river and blocker and eq < .30:
                bluff = .12
            elif ip and not current and eq < .45 and not wet:
                bluff = .13
        if bluff and self.rng.random() < bluff * self.config['bluff_scale']:
            return self._raise(o, hero['bet'] + max(bb, pot * (.40 if not river else .66)))
        return self._pass(o)
