"""bot_19: bounded offline poker policy, using the local INTERFACE.md contract.

This is an independently built, configurable policy, not a solved equilibrium
or an implementation of an unavailable official Agent interface.
"""

import random
from poker_eval import features
from icm import call_threshold

# Imported eagerly: file IO and model construction never occur in act().
try:
    from equity_model import predict_from_features, preflop_equity, preflop_percentile, multiway_equity
except ImportError:
    # Development fallback; the distribution validator requires real assets.
    def preflop_equity(hole, range_width=1.0, opponents=1):
        a, b = sorted((c // 4 + 2 for c in hole), reverse=True)
        s = 0.31 + (a + b) * 0.013 + 0.035 * (hole[0] % 4 == hole[1] % 4)
        if a == b:
            s = 0.48 + a * 0.026
        s -= 0.16 * (1 - range_width)
        return max(.02, min(.95, s)) ** (max(1, opponents) ** .88)

    def preflop_percentile(hole):
        return max(.005, min(.99, 1.8 - 2.4 * preflop_equity(hole)))

    def predict_from_features(f):
        base = [.22, .58, .76, .86, .91, .94, .98, .997, 1.][int(f[6])]
        base += .11 * (f[22] == 4) + .055 * (f[25] == 4)
        if f[6] == 1:
            base += .015 * (f[7] - f[12])
        return max(.01, min(.999, base - .10 * f[-1] - .08 * (1 - f[-2])))

    def multiway_equity(hu, f, opponents):
        return hu ** (opponents ** .85)


DEFAULTS = {
    'open_scale': 1.0,
    'call_margin': .005,
    'value_threshold': .60,
    'bluff_scale': .25,
    'range_tightness': 1.0,
    'realization': .92,
    'push_margin': .0,
    'adapt': True,
    'icm': True,
}


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class Agent:
    """One instance per anonymous table. Only public within-table learning."""

    def __init__(self, seed=19, params=None):
        self.initial_seed = seed
        self.params = dict(DEFAULTS)
        if params:
            unknown = set(params) - set(DEFAULTS)
            if unknown:
                raise ValueError('unknown parameters: ' + repr(unknown))
            self.params.update(params)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is None:
            seed = self.initial_seed
        else:
            self.initial_seed = seed
        self.rng = random.Random(seed)
        self.hand_id = None
        self.history_seen = 0
        self.profiles = [[0., 0., 0., 0., 0., 0.] for _ in range(8)]
        self.pre_actions = {}
        self.post_actions = {}
        self.pre_raises = 0
        self.pre_aggressor = None
        self.hand_voluntary = set()
        self.hand_raised = set()
        self.fallbacks = 0
        self.decisions = 0

    def _learn(self, o):
        hand_id = o['hand_id']
        if hand_id != self.hand_id:
            self.hand_id = hand_id
            self.history_seen = 0
            self.pre_actions = {}
            self.post_actions = {}
            self.pre_raises = 0
            self.pre_aggressor = None
            self.hand_voluntary = set()
            self.hand_raised = set()
        history = o.get('history', ())
        # A full public history is the documented contract. A truncated adapter
        # can supply an empty history; no hidden or cross-table data is inferred.
        if len(history) < self.history_seen:
            self.history_seen = 0
        for ev in history[self.history_seen:]:
            action = ev.get('action')
            seat = ev.get('seat', -1)
            if not 0 <= seat < 8 or action not in ('fold', 'check', 'call', 'raise'):
                continue
            prof = self.profiles[seat]
            if ev.get('street') == 'preflop':
                if seat not in self.pre_actions:
                    prof[0] += 1  # observed first decisions
                if action in ('call', 'raise') and seat not in self.hand_voluntary:
                    prof[1] += 1
                    self.hand_voluntary.add(seat)
                if action == 'raise':
                    if seat not in self.hand_raised:
                        prof[2] += 1
                        self.hand_raised.add(seat)
                    self.pre_raises += 1
                    self.pre_aggressor = seat
                self.pre_actions[seat] = ev
            else:
                if ev.get('to_call_before', 0) > 0:
                    prof[3] += 1  # observed faced-bet decisions
                    prof[4] += action == 'fold'
                prof[5] += action == 'raise'
                if action in ('raise', 'call'):
                    old = self.post_actions.get(seat, (0, 0, 0.))
                    self.post_actions[seat] = (old[0] + (action == 'raise'),
                                               old[1] + (action == 'call'),
                                               ev.get('amount', 0) / max(1, ev.get('pot_before', 1)))
        self.history_seen = len(history)

    def _profile(self, seat):
        if not self.params['adapt']:
            return .30, .18, .45
        p = self.profiles[seat]
        # Shrink toward broad priors; one surprising action cannot dominate.
        vpip = (p[1] + 7.2) / (p[0] + 24.)
        pfr = (p[2] + 4.32) / (p[0] + 24.)
        fold = (p[4] + 8.1) / (p[3] + 18.)
        return _clip(vpip, .08, .90), _clip(pfr, .025, .70), _clip(fold, .08, .88)

    def _ranges(self, o, villains):
        widths, tilts, folds = [], [], []
        bb = o['big_blind']
        for v in villains:
            vpip, pfr, fold = self._profile(v)
            ev = self.pre_actions.get(v)
            width = 1.0
            if ev:
                if ev['action'] == 'raise' or v in self.hand_raised:
                    depth = ev.get('stack_before', o['stacks'][v]) / max(1, bb)
                    size = ev.get('total', bb) / max(1, bb)
                    width = pfr * 1.20
                    if self.pre_raises >= 2:
                        width *= .48 ** min(2, self.pre_raises - 1)
                    if size > 8 and depth > 22:
                        width *= .6
                    if depth <= 12:
                        width = max(width, .17 + .27 / (1 + depth / 4))
                elif ev['action'] == 'call':
                    width = max(.16, vpip * (1.2 if self.pre_raises == 0 else .9))
            tilt = 0.
            if v in self.post_actions:
                raises, calls, ratio = self.post_actions[v]
                tilt = .20 * raises + .08 * calls + .13 * min(2., ratio)
                if pfr > .28:
                    tilt *= .72
                elif pfr < .11:
                    tilt *= 1.20
            widths.append(_clip(width * self.params['range_tightness'], .05, 1.))
            tilts.append(_clip(tilt, 0., .90))
            folds.append(fold)
        return widths, tilts, folds

    def act(self, observation):
        """Bounded deterministic work, no IO/network, legal output or check/fold."""
        self.decisions += 1
        try:
            o = observation
            if len(o['hole']) != 2 or len(o['stacks']) > 8:
                raise ValueError('invalid local observation')
            self._learn(o)
            action = self._decide(o)
            return self._legalize(action, o)
        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError, ArithmeticError):
            self.fallbacks += 1
            return {'action': 'check' if 'check' in observation.get('legal_actions', ()) else 'fold', 'amount': 0}

    @staticmethod
    def _legalize(action, o):
        kind, amount = action
        legal = o['legal_actions']
        if kind == 'raise':
            if 'raise' not in legal or not o.get('can_raise', True):
                kind = 'check' if 'check' in legal else 'call'
            else:
                maximum = int(o['max_raise_to'])
                minimum = int(o['min_raise_to'])
                amount = min(maximum, max(minimum, int(amount)))
                if amount > o['bets'][o['hero']] + o['to_call']:
                    return {'action': 'raise', 'amount': amount}
                kind = 'call'
        if kind not in legal:
            kind = 'check' if 'check' in legal else 'fold'
        return {'action': kind, 'amount': 0}

    def _decide(self, o):
        h = o['hero']
        villains = [i for i, f in enumerate(o['folded']) if i != h and not f]
        if not villains:
            return ('check', 0)
        call = min(o['to_call'], o['stacks'][h])
        cap = o['contributions'][h] + call
        eligible_pot = sum(min(c, cap) for c in o['contributions'])
        # With no wager, all existing pot layers up to hero's available stack
        # are contestable; using the current cap alone would erase a free pot.
        if not call:
            cap = o['contributions'][h] + o['stacks'][h]
            eligible_pot = sum(min(c, cap) for c in o['contributions'])
        odds = call / max(1, eligible_pot + call)
        if self.params['icm'] and call and len(villains) == 1:
            odds = call_threshold(o, villains[0], odds)
        widths, tilts, folds = self._ranges(o, villains)
        if not o['board']:
            return self._preflop(o, villains, widths, folds, call, eligible_pot, odds)
        return self._postflop(o, villains, widths, tilts, folds, call, eligible_pot, odds)

    def _preflop(self, o, villains, widths, folds, call, pot, odds):
        h, bb = o['hero'], max(1, o['big_blind'])
        stack, bet = o['stacks'][h], o['bets'][h]
        depth = (stack + bet) / bb
        n = len(villains)
        hole = o['hole']
        pct = preflop_percentile(hole)
        high, low = sorted((c // 4 + 2 for c in hole), reverse=True)
        suited = hole[0] % 4 == hole[1] % 4
        pair = high == low
        # Seat order is measured relative to button, independent of dead seats.
        size = len(o['stacks'])
        before_button = (o['button'] - h) % size
        behind = sum(0 < (v-h) % size <= before_button for v in villains)
        live = sum(s + c > 0 for s, c in zip(o['stacks'], o['contributions']))
        first_in = self.pre_raises == 0
        limpers = sum(ev['action'] == 'call' for s, ev in self.pre_actions.items() if s != h)
        opponents_committed = [v for v in villains if o['bets'][v] > bb]
        width = min(widths) if opponents_committed else sum(widths) / n

        # A forced all-in blind is not an unopened strategic raising spot.
        # Price the actual showdown before any position/opening-range filter.
        if all(o['stacks'][v] == 0 for v in villains):
            eq = preflop_equity(hole, width, n)
            return ('call' if call and eq >= odds else 'fold' if call else 'check', 0)

        if first_in:
            if h == o['button']:
                opening = .53 if live > 2 else .80
            elif h == o.get('small_blind_seat'):
                opening = .58
            elif before_button == 0:
                opening = .50
            elif behind == 0:
                opening = .58  # small blind / late dead-button seats
            else:
                opening = .46 / (1 + .23 * behind)
            opening *= self.params['open_scale']
            opening *= 1 - .12 * min(3, limpers)
            # Suited connected holdings realize their equity more reliably.
            adjusted = pct - (.025 if suited and high-low <= 3 else 0)
            if depth <= 13 and 'raise' in o['legal_actions']:
                if adjusted < max(.2, opening * 1.3) or depth < 5:
                    ev = self._push_ev(o, villains, hole, pot)
                    if ev > bb * self.params['push_margin']:
                        return ('raise', o['max_raise_to'])
                if depth < 8:
                    return ('check' if not call else 'fold', 0)
            if adjusted < opening:
                amount = bb * (2.25 + .85 * limpers)
                if max(self._profile(v)[0] for v in villains) > .55 and pct < .10:
                    amount += bb
                if amount - bet > .30 * stack:
                    amount = o['max_raise_to']
                return ('raise', amount)
            if not call:
                return ('check', 0)
            # Pot odds for cheap completes/overlimps, with realization discount.
            eq = preflop_equity(hole, max(.4, width), max(1, min(n, limpers + 1)))
            if limpers and call <= bb and eq * .70 > odds + .06 and (pair or suited or high >= 12):
                return ('call', 0)
            return ('fold', 0)

        # Count only opponents who have put money in beyond their blind when
        # pricing a raised pot; players yet to act are separately penalized.
        committed = max(1, len(opponents_committed))
        eq = preflop_equity(hole, width, committed)
        if call >= .90 * stack or all(o['stacks'][v] == 0 for v in villains):
            return ('call' if eq >= odds + .008 else 'fold', 0)
        # Strong hands punish opens; deep 4-bets are deliberately narrower.
        premium = (pair and high >= (11 if self.pre_raises == 1 else 13)) or (high == 14 and low == 13 and self.pre_raises <= 2)
        if premium and 'raise' in o['legal_actions']:
            if depth < 24 or self.pre_raises >= 3:
                if eq > .49 or (pair and high >= 13):
                    return ('raise', o['max_raise_to'])
            else:
                target = max(o['bets']) * (3.0 if behind == 0 else 3.6)
                if target > .34 * (stack + bet):
                    target = o['max_raise_to']
                return ('raise', target)
        if depth <= 15 and 'raise' in o['legal_actions'] and eq > max(.48, odds + .12):
            return ('raise', o['max_raise_to'])
        realization = (.86 if behind == 0 else .75) + .05 * suited
        if pair:
            realization += .03
        # Future domination and squeeze risk must not vanish just because the
        # immediate pot odds of an open are attractive.
        tax = .035 + .012 * min(3, behind) + .02 * max(0, self.pre_raises - 1)
        if call > .20 * stack:
            realization = .94
            tax += .025
        if eq * realization > odds + tax + self.params['call_margin']:
            # Enforce a plausible continuation range against a small open.
            continuation = _clip(width * 1.8 + .08 * suited + .07 * pair, .12, .60)
            if pct < continuation or call < bb:
                return ('call', 0)
        # Set mining is only considered when the effective reward is large.
        effective = max(min(stack, o['stacks'][v]) for v in villains)
        if pair and call <= effective / 18 and self.pre_raises == 1:
            return ('call', 0)
        return ('check' if not call else 'fold', 0)

    def _push_ev(self, o, villains, hole, pot):
        h = o['hero']
        bb = max(1, o['big_blind'])
        risk = o['stacks'][h]
        if risk <= 0:
            return -1e9
        probs, widths = [], []
        for v in villains:
            vpip, pfr, _ = self._profile(v)
            depth = min(o['max_raise_to'], o['stacks'][v]+o['bets'][v]) / bb
            # Caller ranges tighten with shove depth, widen against tiny stacks.
            base = _clip(.11 + .60 / (depth + 1.5), .10, .65)
            rate = _clip(base * (.65 + 1.2 * vpip), .06, .75)
            if o['stacks'][v] == 0:
                rate = 1.
            probs.append(rate)
            widths.append(_clip(rate, .05, 1.))
        # Enumerate at most128 caller subsets. Side-pot chips are assigned to
        # the actual contesting subset at each contribution level. All-in
        # opponents add no new chips; unmatched hero chips return with equity1.
        # Range choices are approximations; chip accounting here is exact.
        count = 1 << len(villains)
        mass = [0.] * count
        mass[0] = 1.
        for j,p in enumerate(probs):
            bit = 1 << j
            for mask in range(bit):
                old = mass[mask]
                mass[mask | bit] = old*p
                mass[mask] = old*(1-p)
        old = o['contributions']
        hero_final = old[h]+risk
        finals = [old[v] + min(o['stacks'][v], max(0,o['max_raise_to']-o['bets'][v])) for v in villains]
        levels = sorted(set(old + finals + [hero_final]))
        layers = []
        previous = 0
        for level in levels:
            if level <= 0 or level > hero_final:
                continue
            base = 1 + sum(c >= level for i,c in enumerate(old) if i != h)
            added = eligible = 0
            for j,v in enumerate(villains):
                if finals[j] >= level:
                    eligible |= 1 << j
                    if old[v] < level:
                        added |= 1 << j
            layers.append((level-previous,base,added,eligible))
            previous = level
        weights = [0.] * count
        for mask,p in enumerate(mass):
            if p:
                for delta,base,added,eligible in layers:
                    weights[mask & eligible] += p*delta*(base+(mask & added).bit_count())
        ev = weights[0]-risk
        sums = [0.] * count
        for mask in range(1,count):
            bit = mask & -mask
            sums[mask] = sums[mask ^ bit] + widths[bit.bit_length()-1]
            if weights[mask]:
                n = mask.bit_count()
                ev += weights[mask]*preflop_equity(hole,sums[mask]/n,n)
        return ev

    def _postflop(self, o, villains, widths, tilts, folds, call, pot, odds):
        h = o['hero']
        n = len(villains)
        stack = o['stacks'][h]
        board = o['board']
        street = len(board)
        width = .65 * min(widths) + .35 * sum(widths) / n
        tilt = .60 * max(tilts) + .40 * sum(tilts) / n
        f = features(o['hole'], board, width, tilt)
        hu = _clip(predict_from_features(f), .001, .999)
        eq = multiway_equity(hu, f, n)
        # Universally unbeatable shared boards must retain exact split equity.
        shared_nuts = street == 5 and f[38] == 0 and (
            (f[34] == 8 and f[35] == 14) or
            (f[34] == 4 and f[35] == 14 and f[21] <= 2) or
            (f[34] == 7 and (f[36] == 14 or (f[35] == 14 and f[36] == 13))))
        if shared_nuts:
            return ('call' if call else 'check', 0)
        # Private royal/straight flush is essentially immune to model noise.
        if f[6] == 8 and f[7] == 14:
            hu = eq = 1.
        respond = [v for v in villains if o['stacks'][v] > 0]
        if not respond:
            return ('call' if call and eq >= odds else 'fold' if call else 'check', 0)
        size = len(o['stacks'])
        last = max(villains + [h], key=lambda s: (s-o['button']-1) % size)
        in_position = last == h
        draw = street < 5 and ((f[22] == 4 and f[23] > 0) or (f[25] == 4 and f[39] > 0))
        wet = f[21] >= 2 and f[26] >= 3
        spr = min(stack, max(o['stacks'][v] for v in respond)) / max(1, pot)
        realized = 1. if street == 5 or call >= .8*stack or spr < .6 else self.params['realization']
        if not in_position and street < 5:
            realized -= .04
        if n > 1 and street < 5:
            realized -= .03
        margin = self.params['call_margin'] * (1 if street < 5 else .4)
        if call:
            # Repeated raises deserve range tightening; the model already sees
            # those raises through tilt, so avoid an unbounded second penalty.
            if eq * realized < odds + margin:
                return ('fold', 0)
            passive = ('call', 0)
        else:
            passive = ('check', 0)
        if 'raise' not in o['legal_actions']:
            return passive
        # Risk only as much as the largest live opponent can match.
        maxmatch = max(o['stacks'][v] + o['bets'][v] for v in respond)
        maximum = min(o['max_raise_to'], maxmatch)
        if maximum <= max(o['bets']):
            return passive
        # Strong hands bet for value, with conservative multiway thresholds.
        value = hu > self.params['value_threshold'] + .065 * min(3, n-1) + (.075 if call else 0)
        nuts = hu > .93 and eq > .72
        low_showdown = hu < .43
        blocker = f[32] >= 13 or (f[1] == 14 and f[29] == 0)
        bluff_ok = (draw or (low_showdown and (blocker or not wet))) and n <= 2 and len(respond) == n
        if call:
            bluff_ok = draw and hu > .38 and n == 1 and spr > 1.4
        if not value and not bluff_ok:
            return passive
        if not value:
            chance = (.42 if draw else .26) * self.params['bluff_scale']
            if not in_position and self.pre_aggressor != h:
                chance *= .65
            if self.rng.random() > chance:
                return passive
        elif not nuts and not call and self.rng.random() < .10:
            return passive  # protect checking range
        best = None
        best_gain = -1e20
        base = max(1, pot + call)
        for frac in (.36, .68, 1.0):
            target = max(o['min_raise_to'], max(o['bets']) + int(base * frac))
            target = min(maximum, target)
            risk = target - o['bets'][h]
            if risk <= call:
                continue
            if risk > .60 * stack:
                target = maximum
                risk = target - o['bets'][h]
            added = max(0, risk - call)
            ratio = added / base
            fs = []
            for v, fold in zip(villains, folds):
                if o['stacks'][v] <= 0:
                    fs.append(0.)
                else:
                    fv = fold + .12 * (ratio - .5) - (.18 if call else 0)
                    fs.append(_clip(fv, .06, .85))
            allfold = 1.
            for fv in fs:
                allfold *= fv
            meanfold = sum(fs) / n
            conditional_hu = _clip((hu - .88 * meanfold) / max(.15, 1-meanfold), .01, .999)
            expected_callers = max(1., n * (1-meanfold))
            called_eq = conditional_hu ** (expected_callers ** .85)
            # Calling ranges are stronger; discount unrealized turn/flop equity.
            if street < 5 and not nuts:
                called_eq *= .95
            extra = sum((1-fv) * min(max(0, target-o['bets'][v]), o['stacks'][v]) for v, fv in zip(villains, fs))
            if 1-allfold > .01:
                extra /= (1-allfold)
            raise_ev = allfold * pot + (1-allfold) * (called_eq*(pot+risk+extra)-risk)
            passive_ev = max(0., eq*realized*(pot+call)-call)
            # Positional checking realizes less than an all-in showdown; still
            # demand an edge rather than betting every marginal made hand.
            gain = raise_ev - passive_ev
            if value and not nuts:
                gain += .025 * pot
            if gain > best_gain:
                best_gain, best = gain, target
        if best is not None and best_gain > (.005 * pot if value else .015 * pot):
            return ('raise', best)
        return passive


if __name__ == '__main__':
    # Optional local JSON-lines adapter. The contest timing boundary is unknown;
    # process startup / JSON IO are not included in our in-process act benchmark.
    import json
    import sys
    agent = Agent()
    for line in sys.stdin:
        obj = json.loads(line)
        if obj.get('reset'):
            agent.reset(obj.get('seed', 19))
            print(json.dumps({'reset': True}), flush=True)
        else:
            print(json.dumps(agent.act(obj)), flush=True)
