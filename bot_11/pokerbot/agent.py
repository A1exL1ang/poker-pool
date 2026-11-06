"""Independent offline NLHE policy using a documented local observation schema.

No official starter/interface was supplied. Agent.act accepts the schema in
README.md. Raise amounts are total street commitments, never added chips.
"""
import math

from .equity import EquityModel, hand_key, load_preflop
from .features import features
from .evaluator import evaluate, parse_card
from .icm import call_threshold

_MODEL = None
_PREFLOP = None


def load_assets():
    """Call during module/agent setup, outside the decision budget."""
    global _MODEL, _PREFLOP
    if _MODEL is None:
        _MODEL = EquityModel()
        _PREFLOP = load_preflop()


def _lerp_range(fraction, values, knots):
    for j in range(len(knots) - 1):
        if fraction >= knots[j + 1]:
            w = (fraction - knots[j + 1]) / (knots[j] - knots[j + 1])
            return values[j] * w + values[j + 1] * (1 - w)
    return values[-1]


class Agent:
    def __init__(self, seed=110011, config=None):
        load_assets()
        self.rng = seed or 110011
        self._initial_seed = self.rng
        self.config = {'open_scale': 1.0, 'call_margin': 0.015, 'value_threshold': 0.62,
                       'bluff_scale': 1.0, 'range_weight': 1.0, 'push_scale': 1.0}
        if config:
            self.config.update(config)
        self.diagnostics = {'decisions': 0, 'errors': 0, 'cache_hits': 0}
        self.stats = {}
        self._hand = None
        self._seen = 0
        self._cache = {}
        self._vpip_seen = set()
        self._pfr_seen = set()
        self._observed_seats = set()

    def _random(self):
        x = self.rng
        x ^= (x << 13) & 0xffffffff
        x ^= x >> 17
        x ^= (x << 5) & 0xffffffff
        self.rng = x & 0xffffffff
        return self.rng / 4294967296.0

    def reset(self):
        """Table boundary reset. The host may also construct a fresh Agent."""
        self.stats.clear()
        self.rng = self._initial_seed
        self._hand = None
        self._seen = 0
        self._cache.clear()
        self._vpip_seen.clear()
        self._pfr_seen.clear()
        self._observed_seats.clear()

    def act(self, obs):
        self.diagnostics['decisions'] += 1
        try:
            return self._act(obs)
        except Exception:
            self.diagnostics['errors'] += 1
            legal = obs.get('legal_actions', ()) if isinstance(obs, dict) else ()
            return {'action': 'check' if 'check' in legal else 'fold'}

    __call__ = act

    def _observe(self, obs, players):
        hand = obs['hand_number']
        if hand != self._hand:
            self._hand, self._seen = hand, 0
            self._cache.clear()
            self._vpip_seen.clear()
            self._pfr_seen.clear()
            self._observed_seats.clear()
        history = obs.get('history', ())
        # Bounded processing; history must contain only the current hand.
        for event in history[self._seen:self._seen + 160]:
            s = self.stats.setdefault(event['seat'], [8.0, 3.5, 1.5, 2.0, 2.0])
            action = event['action']
            if event['street'] == 'preflop':
                if action in ('raise', 'bet', 'call', 'fold', 'check') and event['seat'] not in self._observed_seats:
                    s[0] += 1
                    self._observed_seats.add(event['seat'])
                if action in ('raise', 'bet', 'call') and event['seat'] not in self._vpip_seen:
                    s[1] += 1
                    self._vpip_seen.add(event['seat'])
                if action in ('raise', 'bet') and event['seat'] not in self._pfr_seen:
                    s[2] += 1
                    self._pfr_seen.add(event['seat'])
            elif action == 'fold':
                s[3] += 1
            elif action in ('call', 'raise', 'bet'):
                s[4] += 1
        self._seen = len(history)

    def _act(self, o):
        players = o['players']
        seat = o['seat']
        hero = next(p for p in players if p['seat'] == seat)
        legal = o['legal_actions']
        if not legal:
            raise ValueError('no legal actions')
        hole = tuple(parse_card(c) for c in o['hole_cards'])
        board = tuple(parse_card(c) for c in o['board'])
        if len(hole) != 2 or len(board) not in (0, 3, 4, 5) or len(set(hole + board)) != len(hole + board):
            raise ValueError('invalid cards')
        if {0:'preflop', 3:'flop', 4:'turn', 5:'river'}[len(board)] != o['street']:
            raise ValueError('street/card mismatch')
        self._observe(o, players)
        opponents = [p for p in players if p['seat'] != seat and not p['folded'] and p['stack'] + p['committed'] > 0]
        if not opponents:
            return self._safe(legal)
        call = max(0, min(hero['stack'], o['to_call']))
        bb = max(1, o['big_blind'])
        # Cap inaccessible side pots and uncalled overbets at our total investment.
        cap = hero['committed'] + call
        pot = sum(min(p['committed'], cap) for p in players)
        if cap <= 0:
            pot = o['pot']
        odds = call / max(1, pot + call)
        risk = 0.0
        if o.get('payouts') and len(opponents) == 1 and pot == o['pot'] and call >= 0.3 * max(1, hero['stack']):
            # Only apply the explicit ICM resolution model to heads-up pots with
            # no folded player's larger, inaccessible side pot.
            stacks = [p['stack'] for p in players]
            hidx = players.index(hero)
            vidx = players.index(opponents[0])
            threshold = call_threshold(stacks, hidx, vidx, pot, call, o['payouts'])
            risk = max(0.0, threshold - odds)
        if not board:
            action = self._preflop(o, hero, opponents, hole, call, pot, odds, risk, bb)
        else:
            action = self._postflop(o, hero, opponents, hole, board, call, pot, odds, risk, bb)
        return self._legalize(o, hero, action)

    @staticmethod
    def _safe(legal):
        return {'action': 'check' if 'check' in legal else 'fold'}

    def _legalize(self, o, hero, action):
        kind = action['action']
        legal = o['legal_actions']
        if kind == 'raise':
            if 'raise' not in legal or not o['can_raise']:
                kind = 'call' if o['to_call'] else 'check'
            else:
                maximum = min(o['max_raise_to'], hero['street_bet'] + hero['stack'])
                minimum = min(o['min_raise_to'], maximum)
                amount = max(minimum, min(maximum, int(round(action['amount']))))
                return {'action': 'raise', 'amount': amount}
        if kind == 'call' and not o['to_call']:
            kind = 'check'
        if kind == 'fold' and 'check' in legal:
            kind = 'check'
        return {'action': kind} if kind in legal else self._safe(legal)

    def _pf_fraction(self, o, villain):
        seats = [p['seat'] for p in o['players'] if p['stack'] + p['committed'] > 0]
        bbseat = o.get('big_blind_seat')
        if bbseat is None:
            clockwise = sorted(seats, key=lambda s: (s - o['button'] - 1) % len(o['players']))
            bbseat = clockwise[1] if len(seats) > 2 else next(s for s in seats if s != o['button'])
        ordered = sorted(seats, key=lambda s: (s - bbseat - 1) % len(o['players']))
        nbehind = len(ordered) - ordered.index(villain) - 1
        opening = {0:.38, 1:.55, 2:.43, 3:.31, 4:.245, 5:.20, 6:.165, 7:.14}.get(nbehind, .14)
        if len(seats) == 2:
            opening = .65
        raises = 0
        fraction = 1.0
        for ev in o.get('history', ())[:100]:
            if ev['street'] != 'preflop':
                continue
            if ev['action'] in ('raise', 'bet'):
                raises += 1
                if ev['seat'] == villain:
                    fraction = opening if raises == 1 else (0.055 if raises == 2 else 0.025)
            elif ev['seat'] == villain and ev['action'] == 'call':
                fraction = min(fraction, 0.65 if raises == 0 else (0.40 if raises == 1 else 0.13))
        s = self.stats.get(villain)
        if s and s[0] > 20:
            vpip = min(0.85, s[1] / s[0])
            fraction *= max(0.7, min(1.55, vpip / 0.42))
        return max(0.025, min(1.0, fraction))

    def _preflop(self, o, hero, opponents, hole, call, pot, odds, risk, bb):
        row = _PREFLOP[hand_key(hole)]
        q = row['percentile']
        h, l = sorted((c // 4 + 2 for c in hole), reverse=True)
        suited = hole[0] & 3 == hole[1] & 3
        active = [p for p in o['players'] if p['stack'] + p['committed'] > 0]
        n = len(o['players'])
        # In standard preflop order, button, SB and BB act after nonblind seats.
        clockwise = sorted((p['seat'] for p in active), key=lambda s: (s - o['button'] - 1) % n)
        if len(clockwise) == 2:
            order = [o['button'], next(s for s in clockwise if s != o['button'])]
        elif 'big_blind_seat' in o:
            order = sorted(clockwise, key=lambda s: (s-o['big_blind_seat']-1) % n)
        else:
            order = clockwise[2:] + clockwise[:2]
        behind = len(order) - order.index(o['seat']) - 1
        history = o.get('history', ())
        raises = [ev for ev in history[:100] if ev['street'] == 'preflop' and ev['action'] in ('raise', 'bet')]
        limpers = len({ev['seat'] for ev in history[:100] if ev['street'] == 'preflop' and ev['action'] == 'call'})
        effective = min(hero['stack'] + hero['street_bet'], max(p['stack'] + p['street_bet'] for p in opponents)) / bb
        canraise = o['can_raise'] and 'raise' in o['legal_actions']
        if call and all(p['all_in'] for p in opponents) and not raises:
            eq = row['equity'][0] if len(opponents) == 1 else row['uniform2'] if len(opponents) == 2 else row['uniform3'] ** (len(opponents)/3)
            return {'action': 'call' if eq >= odds + risk else 'fold'}
        if not raises:
            open_fraction = {0: 0.38, 1: 0.55, 2: 0.43, 3: 0.31, 4: 0.245, 5: 0.20, 6: 0.165, 7: 0.14}.get(behind, 0.14)
            if len(active) == 2 and o['seat'] == o['button']:
                open_fraction = 0.78
            open_fraction *= self.config['open_scale']
            if effective <= 12 and canraise:
                # Chip-EV open-shove model. Every possible caller is charged a
                # range-dependent call probability; multiple calls are penalized.
                relevant = [p for p in opponents if p['stack'] > 0]
                prob_no_call = 1.0
                eqsum = weights = calls = 0.0
                riskchips = min(hero['stack'], max(p['stack'] for p in opponents))
                for p in relevant:
                    depth = min(hero['stack'] + hero['street_bet'], p['stack'] + p['street_bet']) / bb
                    fraction = min(0.55, 0.075 + 0.65 / max(1.5, depth))
                    if len(active) <= 3:
                        fraction *= 1.35
                    if p['street_bet'] >= bb:
                        fraction *= 1.15
                    eq = _lerp_range(fraction, row['equity'], (1., .6, .4, .25, .15, .08, .04))
                    prob_no_call *= 1 - fraction
                    eqsum += fraction * eq
                    weights += fraction
                    calls += fraction
                equity = eqsum / max(.001, weights)
                equity *= 1 - max(0., calls - .25) * .13
                equity = max(0., equity - risk)
                ev = prob_no_call * o['pot'] + (1-prob_no_call) * (equity * (o['pot'] + riskchips) - (1-equity) * riskchips)
                if ev > bb * .10 * self.config['push_scale'] and q < max(.35, open_fraction * 1.35):
                    return {'action': 'raise', 'amount': o['max_raise_to']}
                return {'action': 'check' if not call else 'fold'}
            if canraise and q < open_fraction / (1 + .22 * limpers):
                size = ((2.25 if effective > 25 else 2.1) + 1.0 * limpers) * bb
                if limpers:
                    size += .75 * bb
                return {'action': 'raise', 'amount': size}
            if not call:
                return {'action': 'check'}
            # Cheap multiway set-mining/connected suited hands require room to
            # realize equity. Avoid open limps except a cheap small blind complete.
            speculative = h == l or (suited and h-l <= 3 and h >= 7) or (suited and h == 14)
            if limpers and speculative and call <= .045 * effective * bb and q < .56:
                return {'action': 'call'}
            if len(active) <= 3 and call <= .5 * bb and q < .72:
                return {'action': 'call'}
            return {'action': 'fold'}
        aggressor = raises[-1]['seat']
        fraction = self._pf_fraction(o, aggressor)
        eq = _lerp_range(fraction, row['equity'], (1., .6, .4, .25, .15, .08, .04))
        # Only count opponents who have actually invested in the raised pot.
        callers = sum(p['street_bet'] > bb and p['seat'] != aggressor for p in opponents)
        eq *= .85 ** callers
        allin = call >= .80 * hero['stack'] or all(p['all_in'] for p in opponents)
        if canraise:
            premium = q < (.043 if len(raises) == 1 else .016)
            if len(raises) >= 3:
                premium = h == l and h >= 13
            if len(raises) == 2 and effective <= 25:
                premium = premium or (q < .032 and fraction >= .045)
            if risk > .02:
                premium = premium and eq >= odds + risk + .01
            if premium:
                size = o['max_raise_to'] if effective <= 22 or len(raises) >= 2 else o['current_bet'] * (3.4 if behind else 3.0) + callers * o['current_bet']
                return {'action': 'raise', 'amount': size}
            bluff = len(raises) == 1 and effective >= 40 and suited and h == 14 and 2 <= l <= 5 and callers == 0
            if bluff and self._random() < .17 * self.config['bluff_scale']:
                return {'action': 'raise', 'amount': o['current_bet'] * (3.5 if behind else 3.1)}
        realization = 1.0 if allin else (.91 if behind <= 1 else .80)
        if suited or h == l:
            realization += .04 if not allin else 0
        threshold = odds / realization + risk + self.config['call_margin']
        if not allin and fraction < .1 and h != l and h <= 12 and not suited:
            threshold += .05
        width = min(.68, fraction * 1.7 + (.16 if odds < .22 else .06))
        if eq >= threshold and (allin or q < width):
            return {'action': 'call'}
        # Implied set odds: both effective stacks must cover at least 18 calls.
        if not allin and h == l and effective * bb >= call * 18 and call <= 4.5 * bb and len(raises) == 1:
            return {'action': 'call'}
        return {'action': 'check' if not call else 'fold'}

    def _postflop(self, o, hero, opponents, hole, board, call, pot, odds, risk, bb):
        key = tuple(sorted(hole)) + tuple(sorted(board))
        if key in self._cache:
            x, prediction = self._cache[key]
            self.diagnostics['cache_hits'] += 1
        else:
            x = features(hole, board)
            prediction = _MODEL.predict(x)
            self._cache[key] = x, prediction
        street = o['street']
        history = o.get('history', ())[:160]
        streetbets = [e for e in history if e['street'] == street and e['action'] in ('raise', 'bet')]
        eqs = []
        for p in opponents:
            fraction = self._pf_fraction(o, p['seat'])
            base = _lerp_range(fraction, [prediction[0], prediction[3], prediction[4], prediction[5]], (1., .4, .15, .05))
            theirbets = [e for e in streetbets if e['seat'] == p['seat']]
            previous_calls = any(e['seat'] == p['seat'] and e['street'] not in ('preflop', street) and e['action'] in ('call', 'raise', 'bet') for e in history)
            w = .20 if previous_calls else 0.0
            severe = False
            if theirbets:
                sizing = call / max(bb, pot - call)
                w = .42 + min(.34, sizing * .25)
                severe = len(streetbets) >= 2 or sizing > .85
                # River ranges must retain a bluff component; never treat every
                # bet as a value hand. This is a model, not a solved equilibrium.
                if street == 'river':
                    w = min(w, .68)
            w *= self.config['range_weight']
            idx = (9 if fraction <= .25 else 7) if severe else (8 if fraction <= .6 else 6)
            eqs.append(max(.001, min(.999, base*(1-w) + prediction[idx]*w)))
        known_win = x[6] == 8 and x[7] == 14 and (len(board) < 5 or evaluate(board) != evaluate(hole+board))
        if known_win:
            eqs = [1.0] * len(opponents)
        equity = 1.0 if known_win else self._multi_equity(eqs, prediction)
        # Exact universal board nuts prevent learned-model error on forced chops.
        forced_chop = False
        if len(board) == 5:
            bs = evaluate(board)
            bcat, btop, bsecond = bs >> 20, (bs >> 16) & 15, (bs >> 12) & 15
            suits = [sum(c & 3 == s for c in board) for s in range(4)]
            forced_chop = ((bcat == 8 and btop == 14) or
                           (bcat == 7 and (bsecond == 14 or btop == 14 and bsecond == 13)) or
                           (bcat == 4 and btop == 14 and max(suits) <= 2))
            if forced_chop:
                equity = 1.0 / (len(opponents) + 1)
        # Side-pot layers can have fewer opponents than the main pot.
        if call:
            contributions = [(p['seat'], p['committed'] + (call if p['seat'] == o['seat'] else 0), p['folded']) for p in o['players']]
            cap = hero['committed'] + call
            levels = sorted({min(c, cap) for _, c, _ in contributions if c > 0})
            prev = expected = totalpot = 0
            eqmap = {p['seat']: e for p, e in zip(opponents, eqs)}
            for level in levels:
                layer = (level - prev) * sum(c >= level for _, c, _ in contributions)
                rival_eqs = [eqmap[s] for s, c, folded in contributions if c >= level and not folded and s in eqmap]
                share = 1.0 if known_win else (1.0/(len(rival_eqs)+1)) if forced_chop else self._multi_equity(rival_eqs, prediction) if rival_eqs else 1.0
                expected += layer * share
                totalpot += layer
                prev = level
            if totalpot:
                equity = expected / totalpot
            if forced_chop:
                return {'action': 'call' if expected + 1e-9 >= call else 'fold'}
        elif forced_chop:
            return {'action': 'check'}
        nopp = len(opponents)
        position = all((p['seat'] - o['button'] - 1) % len(o['players']) <= (o['seat'] - o['button'] - 1) % len(o['players']) or p['all_in'] for p in opponents)
        spr = hero['stack'] / max(bb, pot)
        cat = x[6]
        flushdraw = x[32] == 4 and x[33] > 0 and len(board) < 5
        straightdraw = x[36] == 4 and x[37] > 0 and len(board) < 5
        draw = flushdraw or straightdraw
        wet = x[21] >= 2 or x[38] >= 3
        if call:
            allin = call >= .85 * hero['stack'] or all(p['all_in'] for p in opponents)
            realize = 1.0 if len(board) == 5 or allin else (.93 if position else .85)
            if draw and call / max(1, pot) < .32:
                realize = min(1., realize + .05)
            if cat >= 3:
                realize = max(realize, .97)
            threshold = odds / realize + risk + self.config['call_margin']
            if o['can_raise'] and 'raise' in o['legal_actions'] and any(p['stack'] > 0 for p in opponents):
                value = equity > (.77 if nopp == 1 else .69) and cat >= 1
                semibluff = nopp == 1 and draw and equity > .48 and spr > 2.2 and len(streetbets) == 1 and self._random() < .10 * self.config['bluff_scale']
                if value or semibluff:
                    size = hero['street_bet'] + call + (pot + call) * (.72 if wet else .58)
                    if hero['stack'] + hero['street_bet'] - size < .30 * max(bb, size) or spr < 1.5:
                        size = o['max_raise_to']
                    return {'action': 'raise', 'amount': size}
            return {'action': 'call' if known_win or equity >= threshold else 'fold'}
        if not o['can_raise'] or not any(p['stack'] > 0 for p in opponents):
            return {'action': 'check'}
        threshold = self.config['value_threshold'] + .02 * max(0, nopp - 1)
        shouldbet = equity > threshold and cat >= 1
        if cat == 0 and len(board) == 3 and nopp == 1 and equity > .69:
            shouldbet = True
        pfraises = [e for e in history if e['street'] == 'preflop' and e['action'] in ('raise', 'bet')]
        pfr = pfraises[-1]['seat'] if pfraises else None
        bluff_prob = 0.
        if nopp == 1:
            if len(board) == 3 and pfr == o['seat'] and not wet:
                bluff_prob = .34
            if draw and .25 < equity < threshold:
                bluff_prob = .25 if flushdraw and straightdraw else .18
            if position and not streetbets and equity < .42:
                bluff_prob = max(bluff_prob, .14 if len(board) < 5 else .07)
            if len(board) == 5 and equity < .22 and x[1] == 14:
                bluff_prob = max(bluff_prob, .10)
            s = self.stats.get(opponents[0]['seat'])
            if s:
                bluff_prob *= max(.35, min(1.7, (s[3] / (s[3]+s[4])) / .45))
        if shouldbet or self._random() < bluff_prob * self.config['bluff_scale']:
            fraction = .70 if wet else .48
            if nopp >= 2:
                fraction = .68
            if equity > .88:
                fraction = .9 if len(board) == 5 else .78
            if not shouldbet and not draw:
                fraction = .33 if len(board) == 3 else .55
            amount = hero['street_bet'] + max(bb, o['pot'] * fraction)
            if shouldbet and spr < 1.4 and equity > .68:
                amount = o['max_raise_to']
            elif hero['street_bet'] + hero['stack'] - amount < .22 * max(bb, amount):
                amount = o['max_raise_to']
            return {'action': 'raise', 'amount': amount}
        return {'action': 'check'}

    @staticmethod
    def _multi_equity(eqs, prediction):
        if not eqs:
            return 1.0
        n = len(eqs)
        if n == 1:
            return eqs[0]
        u = max(.005, prediction[0])
        if n == 2:
            un = min(u, prediction[1])
        else:
            u3 = min(prediction[0], prediction[1], prediction[2])
            un = u3 * (u3 / max(.005, prediction[1])) ** (n - 3)
        correction = max(.65, min(2.5, un / max(.00001, u ** n)))
        return max(0., min(min(eqs), math.prod(eqs) * correction))
