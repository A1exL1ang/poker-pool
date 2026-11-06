"""bot_18: offline no-limit Hold'em policy for the documented LOCAL interface.

Instantiate one Agent per table (or call reset). This is not an official
contest adapter: no official starter kit or interface was supplied.
"""
import random

from equity.model import estimate, estimate_premium, forced_tie, describe
from equity.preflop import preflop_percentile
from policy.icm import call_threshold, shove_gain
from policy.texture import texture_from_features


DEFAULTS = {
    'open_scale': 1.2, 'call_margin': 0.025, 'aggression': 1.6,
    'selection': 1.5, 'shove_margin': 0.03, 'adaptation': 1.0,
    'icm_shove': 1.0,
}


class Agent:
    def __init__(self, seed=None, config=None):
        self.config = dict(DEFAULTS)
        if config:
            self.config.update(config)
        self.seed = seed
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.rng.seed(self.seed)
        self.stats = {}
        self.hand_id = None
        self.history_cursor = 0
        self.seen_preflop = set()
        self.seen_vpip = set()
        self.seen_pfr = set()
        self.errors = 0
        self.decisions = 0

    def _stat(self, seat):
        if seat not in self.stats:
            # Counts, with shrinkage toward a moderate bot prior at use time.
            self.stats[seat] = [0, 0, 0, 0, 0, 0, 0]
        return self.stats[seat]

    def _learn(self, obs):
        hid = obs['hand_id']
        if hid != self.hand_id:
            self.hand_id = hid
            self.history_cursor = 0
            self.seen_preflop.clear()
            self.seen_vpip.clear()
            self.seen_pfr.clear()
        history = obs.get('history', ())
        for a in history[self.history_cursor:]:
            seat, kind = a['seat'], a['kind']
            if seat == obs['hero'] or kind in ('small_blind', 'big_blind', 'blind', 'ante'):
                continue
            s = self._stat(seat)
            if a['street'] == 'preflop':
                if seat not in self.seen_preflop:
                    s[0] += 1
                    self.seen_preflop.add(seat)
                if kind in ('call', 'raise') and seat not in self.seen_vpip:
                    s[1] += 1
                    self.seen_vpip.add(seat)
                if kind == 'raise' and seat not in self.seen_pfr:
                    s[2] += 1
                    self.seen_pfr.add(seat)
            else:
                s[3] += 1
                s[4] += kind == 'raise'
                if kind in ('fold', 'call'):
                    s[5] += 1
                    s[6] += kind == 'fold'
        self.history_cursor = len(history)

    def _profile(self, seat):
        s = self._stat(seat)
        weight = self.config['adaptation']
        vpip = (9 + s[1] * weight) / (30 + s[0] * weight)
        pfr = (5.4 + s[2] * weight) / (30 + s[0] * weight)
        agg = (6 + s[4] * weight) / (20 + s[3] * weight)
        fold = (9 + s[6] * weight) / (20 + s[5] * weight)
        return vpip, pfr, agg, fold

    @staticmethod
    def _pass(obs):
        return {'action': 'check' if obs['to_call'] == 0 else 'fold'}

    @staticmethod
    def _call(obs):
        return {'action': 'check' if obs['to_call'] == 0 else 'call'}

    def _raise(self, obs, target):
        if not obs['can_raise']:
            return self._call(obs)
        maximum = int(obs['max_raise_to'])
        minimum = int(obs['min_raise_to'])
        amount = min(maximum, max(minimum, int(round(target))))
        return {'action': 'raise', 'amount': amount}

    def act(self, obs):
        """Return a legal local-contract action. No I/O or network in this path."""
        self.decisions += 1
        try:
            self._learn(obs)
            if obs['street'] == 'preflop':
                return self._preflop(obs)
            return self._postflop(obs)
        except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError, OverflowError):
            self.errors += 1
            return {'action': 'check' if obs.get('to_call', 1) == 0 else 'fold'}

    def _preflop(self, o):
        hero = o['hero']
        players = o['players']
        me = next(p for p in players if p['seat'] == hero)
        opponents = [p for p in players if p['seat'] != hero and not p['folded'] and p['stack'] + p['invested'] > 0]
        alive = [p for p in players if p['stack'] + p['invested'] > 0]
        n = len(alive)
        bb = max(1, o['big_blind'])
        stack = me['stack']
        effective_total = min(stack + me['bet'], max((p['stack'] + p['bet'] for p in opponents), default=stack))
        effective = max(0, effective_total - me['bet'])
        depth = effective_total / bb
        call = min(stack, o['to_call'])
        pct = preflop_percentile(o['hole'])
        hist = [a for a in o.get('history', ()) if a['street'] == 'preflop']
        raises = [a for a in hist if a['kind'] == 'raise']
        limpers = len({a['seat'] for a in hist if a['kind'] == 'call'})
        # Clockwise players still able to act; seat IDs are the physical seats.
        seats = sorted(p['seat'] for p in alive)
        button = o['button']
        after_button = sorted(seats, key=lambda s: (s - button - 1) % 8)
        sb = next((a['seat'] for a in hist if a['kind'] == 'small_blind'), None)
        bbseat = next((a['seat'] for a in hist if a['kind'] == 'big_blind'), after_button[0] if n == 2 else after_button[1])
        order = sorted(seats, key=lambda s: (s - bbseat - 1) % 8)
        hero_index = order.index(hero)
        behind = [p for p in opponents if order.index(p['seat']) > hero_index]
        if not raises:
            if not any(p['stack'] > 0 for p in opponents):
                eligible = sum(min(p['invested'], me['invested'] + call) for p in players)
                threshold = call / max(1, eligible + call)
                if len(opponents) == 1 and eligible == o['pot'] and o.get('payouts'):
                    threshold = max(threshold, call_threshold(players, hero, opponents[0]['seat'], o['pot'], call, o['payouts']))
                return self._call(o) if call == 0 or estimate(o['hole'], (), 1.0, len(opponents)) > threshold + .01 else self._pass(o)
            if depth <= 11 and o['can_raise'] and all(p['stack'] > 0 for p in opponents):
                # One-caller approximation with a multi-caller penalty. All
                # equities come from offline data; no live rollouts.
                foldprob, eqsum, weight = 1.0, 0.0, 0.0
                callers = []
                for p in opponents:
                    vpip, pfr, _, _ = self._profile(p['seat'])
                    r = min(0.65, max(0.08, 0.10 + 0.8 / max(2, depth) + 0.45 * (vpip - 0.30)))
                    if limpers:
                        r = min(0.85, r + 0.10)
                    w = foldprob * r
                    opponent_eq = estimate(o['hole'], (), r, 1)
                    eqsum += w * opponent_eq
                    callers.append((p['seat'], w, max(0, opponent_eq - 0.015 * max(0, len(opponents) - 1))))
                    weight += w
                    foldprob *= 1 - r
                eq = eqsum / max(weight, 1e-9) - 0.015 * max(0, len(opponents) - 1)
                risk = effective
                reward = o['pot'] + max(0, effective + me['bet'] - max((p['bet'] for p in opponents), default=0))
                ev = foldprob * o['pot'] + (1 - foldprob) * (eq * reward - (1 - eq) * risk)
                if self.config['icm_shove'] and n <= 5 and o.get('payouts'):
                    gain, steal_gain = shove_gain(players, hero, o['pot'], o['payouts'], foldprob, callers)
                    if gain > self.config['shove_margin'] * steal_gain:
                        return self._raise(o, o['max_raise_to'])
                    return self._call(o) if call == 0 else self._pass(o)
                if ev > self.config['shove_margin'] * bb or pct < 0.025:
                    return self._raise(o, o['max_raise_to'])
                return self._call(o) if call == 0 else self._pass(o)
            ranges = [0.70, 0.48, 0.34, 0.27, 0.22, 0.185, 0.16, 0.145]
            opening = ranges[min(7, len(behind))]
            if hero == sb:
                opening = 0.68 if n == 2 else 0.50
            if hero == bbseat:
                opening = 0.27 if limpers else 0.70
            opening *= self.config['open_scale']
            if limpers:
                opening *= 0.78 ** min(3, limpers)
            if pct < opening and o['can_raise']:
                size = (2.2 if depth < 35 else 2.5) * bb + limpers * bb
                if hero == sb and n > 2:
                    size += 0.5 * bb
                if size - me['bet'] > 0.38 * effective:
                    size = o['max_raise_to']
                return self._raise(o, size)
            if call == 0:
                return self._call(o)
            # Cheap, multiway suited/pair overlimps only at sufficient depth.
            a, b = o['hole']
            suited = a % 4 == b % 4
            pair = a // 4 == b // 4
            if limpers and depth >= 20 and call <= bb and (pair or suited and pct < 0.5):
                return self._call(o)
            if hero == sb and n == 2 and pct < 0.88 and call <= bb / 2:
                return self._call(o)
            return self._pass(o)
        raiser = raises[-1]['seat']
        vpip, pfr, _, _ = self._profile(raiser)
        amount = max(p['bet'] for p in players)
        r = min(0.65, max(0.055, pfr * (1.35 if n <= 3 else 1.0)))
        if len(raises) == 2:
            r *= 0.44
        elif len(raises) >= 3:
            r *= 0.22
        # Overshoves are strong by default, but repeated maniacs broaden this.
        if amount / bb > 12:
            r *= max(0.24, min(1.0, 0.24 + 2.0 * (pfr - 0.18)))
        elif amount / bb > 5:
            r *= 0.65
        r = max(0.025, r)
        called = sum(p['bet'] > bb and p['seat'] != raiser for p in opponents)
        eq = estimate(o['hole'], (), r, min(7, 1 + called))
        if r < 0.06 and not called:
            # Distinct premium ranges cover deep shoves and repeated raises
            # that the broad percentile table cannot represent accurately.
            tight_eq = estimate_premium(o['hole'], nutted=len(raises) >= 3 and pfr < 0.23)
            blend = min(1.0, (0.06 - r) / 0.035)
            eq = eq * (1 - blend) + tight_eq * blend
        eligible_pot = sum(min(p['invested'], me['invested'] + call) for p in players)
        threshold = call / max(1, eligible_pot + call)
        allin = call >= stack or not any(p['stack'] > 0 for p in opponents)
        if allin and len(opponents) == 1 and eligible_pot == o['pot'] and o.get('payouts'):
            threshold = max(threshold, call_threshold(players, hero, raiser, o['pot'], call, o['payouts']))
        if allin:
            return self._call(o) if eq > threshold + 0.012 else self._pass(o)
        # 3bet/4bet a compact value range; low-frequency suited-ace blockers.
        reraise_cut = (0.048 if len(raises) == 1 else 0.018) * (1 + max(0, pfr - 0.18) * 2)
        if pct < reraise_cut and eq > 0.48 and o['can_raise']:
            target = amount * (3.0 if hero == button else 3.4) + called * amount
            if depth < 25 or target - me['bet'] > 0.40 * effective:
                target = o['max_raise_to']
            return self._raise(o, target)
        # Equity realization accounts for position, potential re-raises and
        # the danger of a dominated hand before a substantial stack remains.
        inpos = hero == button or all((hero - button - 1) % 8 > (p['seat'] - button - 1) % 8 for p in opponents)
        realization = (0.91 if inpos else 0.80) - min(0.12, 0.025 * len(behind))
        if o['hole'][0] % 4 == o['hole'][1] % 4:
            realization += 0.04
        if call > 0.25 * stack:
            realization += 0.07
        if eq * realization > threshold + self.config['call_margin'] and pct < min(0.55, r * 1.8 + 0.09):
            return self._call(o)
        # Set mining only with enough money behind and a modest call price.
        if o['hole'][0] // 4 == o['hole'][1] // 4 and effective > 18 * call and len(raises) == 1 and call <= 4 * bb:
            return self._call(o)
        return self._pass(o)

    def _postflop(self, o):
        hero = o['hero']
        players = o['players']
        me = next(p for p in players if p['seat'] == hero)
        opp = [p for p in players if p['seat'] != hero and not p['folded'] and p['stack'] + p['invested'] > 0]
        n = len(opp)
        if not n:
            return self._call(o)
        if forced_tie(o['hole'], o['board']):
            # Calling cannot lose money in a guaranteed chop. A binary
            # win/lose ICM approximation is inappropriate for this branch.
            return self._call(o)
        street = o['street']
        hist = o.get('history', ())
        pre = [a for a in hist if a['street'] == 'preflop' and a['kind'] == 'raise']
        now = [a for a in hist if a['street'] == street]
        bets = [a for a in now if a['kind'] == 'raise']
        aggressor = bets[-1]['seat'] if bets else (pre[-1]['seat'] if pre else None)
        villain = next((p for p in opp if p['seat'] == aggressor), max(opp, key=lambda p: p['bet']))
        vpip, pfr, agg, fold = self._profile(villain['seat'])
        r = max(0.07, pfr) if any(a['seat'] == villain['seat'] for a in pre) else max(0.18, vpip)
        if len(pre) >= 2:
            r *= 0.5
        raw = estimate(o['hole'], o['board'], r, n)
        tex = texture_from_features(describe(o['hole'], o['board']))
        call = min(me['stack'], o['to_call'])
        pot = o['pot']
        eligible_pot = sum(min(p['invested'], me['invested'] + call) for p in players)
        effective_total = min(me['stack'] + me['bet'], max((p['stack'] + p['bet'] for p in opp), default=0))
        effective = max(0, effective_total - me['bet'])
        spr = effective / max(1, pot)
        inpos = all((hero - o['button'] - 1) % 8 > (p['seat'] - o['button'] - 1) % 8 for p in opp)
        # Preflop-range equity needs adjustment for postflop action selection.
        # The transform is a heuristic, measured separately from label accuracy.
        prior_bets = sum(a['kind'] == 'raise' and a['seat'] != hero and a['street'] not in ('preflop', street) for a in hist)
        pressure = (0.08 + prior_bets * 0.12) * self.config['selection']
        size = call / max(1, pot - call)
        if call:
            pressure += (0.22 + 0.24 * min(2, size) + 0.24 * max(0, len(bets) - 1)) * self.config['selection']
            if street == 'river':
                pressure += 0.16 * self.config['selection']
            pressure *= max(0.35, min(1.6, 1.45 - 1.5 * agg))
        eq = raw ** (1 + pressure)
        margin = self.config['call_margin']
        if call:
            if raw >= 1.0:
                # Exact certain wins (e.g. hero's royal flush) must not be
                # rejected by an uncertainty margin near an ICM boundary.
                if o['can_raise']:
                    return self._raise(o, me['bet'] + call + 0.8 * (pot + call))
                return self._call(o)
            threshold = call / max(1, eligible_pot + call)
            terminal = street == 'river' or call >= me['stack'] or not any(p['stack'] for p in opp)
            if terminal and n == 1 and o.get('payouts') and eligible_pot == pot and call > 0.15 * me['stack']:
                threshold = max(threshold, call_threshold(players, hero, villain['seat'], pot, call, o['payouts']))
            # Nonterminal equity realization, modest credit for robust draws.
            realization = 1.0 if terminal else (0.94 if inpos else 0.87)
            if tex['draw'] and not terminal:
                realization += 0.05
            if eq * realization <= threshold + margin:
                return self._pass(o)
            raise_eq = 0.82 if n == 1 else 0.88
            if eq > raise_eq and o['can_raise'] and (raw > 0.94 or spr < 2.0):
                target = me['bet'] + call + (pot + call) * (0.65 if street != 'river' else 0.85)
                if target - me['bet'] > 0.6 * effective or spr < 1.2:
                    target = o['max_raise_to']
                return self._raise(o, target)
            # Semi-bluffs sparingly: draws alone are not a reason to stack off.
            if n == 1 and street == 'flop' and tex['nutfd'] and eq > 0.48 and fold > 0.48 and spr > 2 and o['can_raise'] and self.rng.random() < 0.12 * self.config['aggression']:
                return self._raise(o, me['bet'] + call + 0.7 * (pot + call))
            return self._call(o)
        if not o['can_raise']:
            return self._call(o)
        ag = self.config['aggression']
        value_cut = (0.64 if street == 'flop' else 0.68 if street == 'turn' else 0.72) + 0.035 * max(0, n - 1)
        # A wide caller pays a thinner value range; a very tight caller doesn't.
        value_cut -= 0.12 * max(0, vpip - 0.30)
        value = eq > value_cut
        cbet = bool(pre and pre[-1]['seat'] == hero)
        checks = sum(a['kind'] == 'check' and a['seat'] != hero for a in now)
        bluff = 0.0
        if n == 1 and raw < 0.55:
            if street == 'flop':
                bluff = (0.28 if cbet else 0.13) * (1.0 if inpos else 0.60)
                if not tex['wet']:
                    bluff *= 1.3
                if tex['draw']:
                    bluff += 0.16
            elif street == 'turn':
                bluff = 0.22 if tex['draw'] else 0.08 if inpos else 0.03
            else:
                bluff = 0.12 if raw < 0.3 and tex['ace_blocker'] else 0.045 if raw < 0.2 and inpos else 0.0
            bluff *= max(0.3, min(1.6, fold / 0.45)) * ag
            if checks and inpos:
                bluff *= 1.25
        elif n == 2 and street == 'flop' and tex['nutfd'] and cbet:
            bluff = 0.12 * ag
        if value or self.rng.random() < bluff:
            fraction = 0.36 if street == 'flop' and not tex['wet'] and n == 1 else 0.58
            if street == 'river' or value and eq > 0.90:
                fraction = 0.78
            # Use the same sizing for corresponding value and bluff branches.
            target = me['bet'] + max(o['big_blind'], fraction * pot)
            if value and (target - me['bet'] > 0.60 * effective or spr < 0.8) and eq > 0.72:
                target = o['max_raise_to']
            return self._raise(o, target)
        return self._call(o)
