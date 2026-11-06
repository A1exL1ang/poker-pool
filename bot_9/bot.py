"""bot_9: independent local NLHE agent. See README for the nonofficial contract.

No network, subprocess, native extension, or training dependency is used here.
Construct once per table, then call act(observation) for each decision.
"""
import math
from model import EquityModel
from evaluator import evaluate

try:
    from equity import preflop_equity, preflop_percentile, equity_vs_range
except ImportError:
    # Runnable checkpoint before offline assets exist.
    def preflop_equity(hole, opponents=1):
        r = sorted((c // 4 + 2 for c in hole), reverse=True)
        e = (.49 + r[0] * .025 if r[0] == r[1] else
             .25 + .018 * r[0] + .008 * r[1] + .035 * (hole[0] % 4 == hole[1] % 4))
        return min(.86, e) ** (opponents ** .72)
    def preflop_percentile(hole):
        return max(0.001, min(.999, (0.76-preflop_equity(hole)) / .40))
    def equity_vs_range(hole, top_fraction):
        e = preflop_equity(hole)
        return max(.12, e - .12 * (1-top_fraction))


DEFAULTS = {
    'open_scale': 1.1, 'call_margin': .05, 'value_threshold': .69,
    'bluff_scale': .5, 'risk_scale': 1.0, 'cbet': .50,
    'postflop_range_weight': .60, 'shove_max_bb': 11.0,
    'conditional_weight': .95, 'bluff_prior': .10, 'bet_penalty_scale': 1.25,
}


def _icm_hero(stacks, payouts, hero):
    """Exact hero ICM via subset probabilities, used only with <=5 survivors."""
    n = len(stacks)
    total = sum(stacks)
    if total <= 0:
        return sum(payouts[:n]) / max(1, n)
    if stacks[hero] <= 0:
        zeros = sum(s <= 0 for s in stacks)
        return sum(payouts[n-zeros:n]) / zeros
    others = [i for i in range(n) if i != hero]
    sums = [0] * (1 << len(others))
    probs = [0.] * len(sums)
    probs[0] = 1.
    answer = 0.
    for mask in range(len(sums)):
        if mask:
            bit = mask & -mask
            sums[mask] = sums[mask ^ bit] + stacks[others[bit.bit_length()-1]]
        remaining = total - sums[mask]
        place = mask.bit_count()
        if remaining <= 0:
            continue
        answer += probs[mask] * stacks[hero] / remaining * payouts[place]
        for j, seat in enumerate(others):
            if not mask & (1 << j):
                probs[mask | (1 << j)] += probs[mask] * stacks[seat] / remaining
    return answer


class Agent:
    def __init__(self, seed=9, params=None, model_path=None):
        self.params = dict(DEFAULTS)
        if params:
            self.params.update(params)
        self.model = EquityModel(model_path)
        self.state = seed & 0xFFFFFFFF or 9
        self.errors = 0
        self._hand = None
        self._history_index = 0
        self._seen_pre = set()
        self.stats = []

    def _random(self):
        # Independent table-local deterministic stream; no reseed per hand.
        x = self.state
        x ^= x << 13 & 0xFFFFFFFF
        x ^= x >> 17
        x ^= x << 5 & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state / 4294967296.

    @staticmethod
    def fallback(obs):
        return {'action': 'check' if obs.get('to_call', 0) == 0 else 'fold'}

    @staticmethod
    def _raise(obs, target):
        if not obs['can_raise']:
            return {'action': 'call'} if obs['to_call'] else {'action': 'check'}
        maximum = obs['max_raise_to']
        target = min(maximum, max(obs['min_raise_to'], int(target)))
        return {'action': 'raise', 'amount': target}

    def act(self, obs):
        """Return {action: fold/check/call/raise, amount?: total street raise-to}.

        Schema checking belongs in the integration adapter, outside the timed
        strategic call. Exceptions still receive a check/fold safety fallback.
        """
        try:
            self._observe(obs)
            if obs['street'] == 0:
                return self._preflop(obs)
            return self._postflop(obs)
        except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError, OverflowError):
            self.errors += 1
            return self.fallback(obs)

    def _observe(self, o):
        """Seat-local, shrunk action frequencies from already public history.

        Only observed actions count; folds can hide the rest of a hand, so this
        is a deliberately conservative signal rather than a full HUD.
        """
        if not self.stats:
            self.stats = [[0, 0, 0, 0, 0] for _ in o['stacks']]
        if self._hand != o['hand_no']:
            self._hand = o['hand_no']
            self._history_index = 0
            self._seen_pre.clear()
        history = o['history']
        for i in range(self._history_index, len(history)):
            a = history[i]
            seat, action = a['seat'], a['action']
            if seat == o['hero']:
                continue
            s = self.stats[seat]
            if a['street'] == 0 and seat not in self._seen_pre:
                self._seen_pre.add(seat)
                s[0] += 1
                s[1] += action in ('call', 'raise')
                s[2] += action == 'raise'
            elif a['street'] > 0:
                s[3] += 1
                s[4] += action == 'raise'
        self._history_index = len(history)

    def _rates(self, seat):
        if seat is None or not self.stats:
            return .27, .17, .29
        n, vpip, pfr, post, bet = self.stats[seat]
        return ((vpip + 6.75) / (n + 25), (pfr + 4.25) / (n + 25),
                (bet + 5.8) / (post + 20))

    def _risk(self, o, cost, villain):
        """Extra call-equity required by the supplied payout curve only.

        Exact ICM binary-outcome premium at <=5 survivors; bounded surrogate
        above that. No payout is guessed when none is provided.
        """
        payouts = o.get('payouts')
        if not payouts or len(payouts) < len(o['stacks']):
            return 0.
        h = o['hero']
        alive = [i for i, v in enumerate(o['stacks']) if v + o['contributions'][i] > 0]
        n = len(alive)
        if n <= 2 or len(set(payouts[:n])) == 1:
            return 0.
        if all(float(v) == 0 for v in payouts[1:n]):
            return 0.
        hs = o['stacks'][h]
        fraction = cost / max(1, hs)
        cap = o['contributions'][h] + cost
        heads_up_pot = sum(not f for f in o['folded']) == 2
        no_sidepot = all(c <= cap for c in o['contributions'])
        if n <= 5 and fraction > .15 and heads_up_pot and no_sidepot:
            s = [o['stacks'][i] for i in alive]
            hi = alive.index(h)
            if villain not in alive:
                return 0.
            vi = alive.index(villain)
            # Approximate heads-up winner takes the existing pot; the call
            # contributes cost once. Every state conserves tournament chips.
            base = list(s)
            pot = o['pot']
            win = list(base)
            lose = list(base)
            amount = min(cost, hs)
            win[hi] += pot
            lose[hi] -= amount
            lose[vi] += pot + amount
            # Hero fold equity is measured against a pot awarded to villain.
            base[vi] += pot
            p = [float(v) for v in payouts[:n]]
            u0 = _icm_hero(base, p, hi)
            uw = _icm_hero(win, p, hi)
            ul = _icm_hero(lose, p, hi)
            chip_threshold = cost / max(1, pot + cost)
            premium = (u0-ul) / max(1e-12, uw-ul) - chip_threshold
            return min(.22, max(0., premium)) * self.params['risk_scale']
        # Early-stage surrogate, small by construction; not an exact ICM solver.
        curvature = sum(payouts[1:n]) / max(1e-9, sum(payouts[:n]))
        short = min(o['stacks'][i] + o['contributions'][i] for i in alive)
        pressure = 1 + .7 * (short < hs * .3)
        return min(.07, .045 * curvature * fraction * pressure) * self.params['risk_scale']

    def _preflop(self, o):
        h, hole, bb = o['hero'], o['hole'], max(1, o['bb'])
        stack, put = o['stacks'][h], o['committed'][h]
        call = min(stack, o['to_call'])
        pot = o['pot']
        contestable = sum(min(c, o['contributions'][h] + call) for c in o['contributions'])
        nseats = len(o['stacks'])
        live = [i for i in range(nseats) if not o['folded'][i]]
        others = [i for i in live if i != h]
        if not others:
            return self.fallback(o)
        active = [i for i in range(nseats) if o['stacks'][i] + o['contributions'][i] > 0]
        # Distance clockwise from hero to button: zero is button; SB/BB handled
        # by actual hand history and committed blinds, never hardcoded seats.
        order = sorted(active, key=lambda i: (i-o['button']-1) % nseats)
        sb = o['button'] if len(active) == 2 else order[0]
        big = next(i for i in order if i != sb) if len(active) == 2 else order[1]
        after = sorted(others, key=lambda i: (i-h) % nseats)
        if h == big:
            behind = 0
        elif h == sb:
            behind = 1
        else:
            behind = sum((i-h) % nseats <= (big-h) % nseats for i in others)
        eff = min(stack + put, max(o['stacks'][i] + o['committed'][i] for i in others)) / bb
        raises = [a for a in o['history'] if a['street'] == 0 and a['action'] == 'raise']
        percentile = preflop_percentile(hole)
        high, low = sorted((c // 4 + 2 for c in hole), reverse=True)
        paired, suited = high == low, hole[0] % 4 == hole[1] % 4
        if not raises:
            limpers = sum(a['action'] == 'call' and a['street'] == 0 for a in o['history'])
            opening = (.62 if behind == 1 else .45 if behind == 2 else
                       .32 if behind == 3 else .25 if behind == 4 else
                       .20 if behind == 5 else .17 if behind == 6 else .15)
            opening *= self.params['open_scale']
            opening /= 1 + .13 * limpers
            if behind == 0:
                opening = .15 / (1 + .15 * max(0, limpers-1))
            if eff <= self.params['shove_max_bb'] and not limpers and behind > 0:
                # Restricted to plausible top hands; estimate all-fold and
                # single-caller chip EV, tightening as more players remain.
                cf = min(.38, max(.10, .07 + .8 / max(2., eff)))
                eq = equity_vs_range(hole, cf)
                risk = min(stack, max(o['stacks'][i] for i in others))
                fe = (1-cf) ** behind
                ev = fe * pot + (1-fe) * (eq * (pot + risk) - (1-eq)*risk)
                cap = min(.85, opening * (1.55 if eff <= 5 else 1.05))
                if percentile < cap and ev > .02 * bb:
                    return self._raise(o, o['max_raise_to'])
                return self.fallback(o)
            if percentile < opening and o['can_raise']:
                size = (2.25 if eff < 35 else 2.5) + .85 * limpers
                if h == sb:
                    size += .3
                target = max(size * bb, o['min_raise_to'])
                if stack < target - put + 2 * bb:
                    target = o['max_raise_to']
                return self._raise(o, target)
            if not call:
                return {'action': 'check'}
            # Complete cheaply with playable hands after multiple limpers.
            if limpers and call <= .5 * bb and percentile < .62:
                return {'action': 'call'}
            return {'action': 'fold'}
        raiser = raises[-1]['seat']
        size = max(o['committed']) / bb
        # Range prior is deliberately broad; observed large reraises tighten.
        top = .24 if len(active) <= 3 else .17
        if raiser == sb or raiser == o['button']:
            top = .38
        if len(raises) >= 2:
            top = .065 if len(raises) == 2 else .025
        if size > 12 and len(raises) == 1:
            top = max(.045, min(top, .075 + .6/max(6., size)))
        vpip, pfr_rate, aggression = self._rates(raiser)
        observed = self.stats[raiser][0] if self.stats else 0
        if len(raises) == 1 and observed >= 6:
            top *= min(2.3, max(.55, pfr_rate/.17))
            top = min(.85, max(.035, top))
        eq = equity_vs_range(hole, top)
        callers = [i for i in others if o['committed'][i] >= max(o['committed']) and i != raiser]
        eq **= 1 + .52 * len(callers)
        risk = self._risk(o, call, raiser) if call else 0.
        odds = call / max(1, contestable + call)
        # Value reraises and selective suited ace blocker bluffs at depth.
        premium = (paired and high >= 12) or (high == 14 and low >= 13)
        strong = (paired and high >= 11) or (high == 14 and low >= 12 and suited)
        kings_plus = paired and high >= 13
        aces = paired and high == 14
        value_reraise = ((len(raises) == 1 and (premium or (strong and eff < 32)))
                        or (len(raises) == 2 and (kings_plus or (premium and eff <= 30 and size <= 12)))
                        or (len(raises) >= 3 and aces))
        if o['can_raise'] and value_reraise:
            target = max(o['committed']) * (3.0 if h != sb and h != big else 3.5)
            if eff <= 24 or len(raises) >= 2:
                target = o['max_raise_to']
            return self._raise(o, target)
        if (o['can_raise'] and len(raises) == 1 and eff > 35 and high == 14
                and suited and low <= 5 and not callers and self._random() < .13):
            return self._raise(o, max(o['committed']) * 3.1)
        if not call:
            return {'action': 'check'}
        if call >= stack * .7 or eff <= 13:
            if eq >= odds + risk + .018 + .012 * max(0, len(raises)-1):
                return {'action': 'call'}
            return {'action': 'fold'}
        # Raw showdown equity overstates the value of weak hands with future
        # betting. Position-specific entry limits guard against that error.
        call_cap = min(.70, top*1.65 + .12) if h == big else min(.50, top*1.10 + .065)
        if h == sb:
            call_cap = min(.42, top + .04)
        if percentile > call_cap and not (paired and len(raises) == 1 and call < stack / 22):
            return {'action': 'fold'}
        realization = .80 if h == big or h == sb else .88
        realization -= .04 * len(callers)
        # Deep small pairs can set-mine, only at a price supported by stacks.
        if paired and call <= min(stack, o['stacks'][raiser]) / 18 and len(raises) <= 2:
            return {'action': 'call'}
        if eq * realization >= odds + self.params['call_margin'] + risk:
            # A dominated weak offsuit ace cannot realize raw preflop equity
            # efficiently in a multiway raised pot.
            if not suited and not paired and high == 14 and low < 9 and size >= 3:
                return {'action': 'fold'}
            return {'action': 'call'}
        return {'action': 'fold'}

    def _postflop(self, o):
        h = o['hero']
        stack, put = o['stacks'][h], o['committed'][h]
        call = min(stack, o['to_call'])
        pot, street = o['pot'], o['street']
        contestable = sum(min(c, o['contributions'][h] + call) for c in o['contributions'])
        opponents = [i for i in range(len(o['stacks'])) if i != h and not o['folded'][i]]
        m = len(opponents)
        if not m:
            return self.fallback(o)
        rank = evaluate(tuple(o['hole']) + tuple(o['board']))
        if rank == (8 << 20) | (14 << 16):
            # A royal flush cannot lose. In particular, never fold a board
            # royal to a large bet because of approximate equity penalties.
            if len(o['board']) == 5 and evaluate(o['board']) == rank:
                return {'action': 'call'} if call else {'action': 'check'}
            return self._raise(o, o['max_raise_to']) if o['can_raise'] else ({'action': 'call'} if call else {'action': 'check'})
        eqs = self.model.predict(o['hole'], o['board'])
        pre = [a for a in o['history'] if a['street'] == 0 and a['action'] == 'raise']
        bets = [a for a in o['history'] if a['street'] == street and a['action'] == 'raise']
        last_bettor = bets[-1]['seat'] if bets else None
        # Range equity prior combines top40 and top15 based on preflop action;
        # this is an equity estimator, not an equilibrium value function.
        prior = eqs[1] if len(pre) <= 1 else .4*eqs[2] + .6*eqs[3]
        w = self.params['postflop_range_weight']
        single = (1-w) * eqs[0] + w * prior
        # Multiway correlation makes q**m too pessimistic; exponent is tuned
        # only on local synthetic opponents and later evaluated held-out.
        equity = max(.001, min(.999, single)) ** (1 + .68*(m-1))
        odds = call / max(1, contestable + call)
        bet_ratio = call / max(1, contestable-call)
        effective = min(stack, max(o['stacks'][i] for i in opponents))
        spr = effective / max(1, pot)
        if call:
            # Selection by a bet is separate from preflop range selection.
            # Increasing size and repeated raises are evidence of strength.
            penalty = .035 + .025 * street + .055 * min(2., bet_ratio)
            penalty += .045 * max(0, len(bets)-1) + .025 * (m-1)
            vpip, pfr_rate, aggression = self._rates(last_bettor)
            samples = self.stats[last_bettor][3] if self.stats and last_bettor is not None else 0
            if len(eqs) >= 6:
                # The last two heads estimate equity against the strongest
                # current made hands, with whole rank ties kept in the range.
                # Mix in random holdings for bluffs; draws are approximated by
                # this mixture rather than silently treated as made value.
                tight = min(1., max(0., (bet_ratio-.4)/.9 + .2*max(0,len(bets)-1)))
                made = (1-tight)*eqs[4] + tight*eqs[5]
                bluff = self.params['bluff_prior']
                if samples >= 8:
                    bluff += .45*(aggression-.29)
                bluff = min(.42, max(.06, bluff))
                conditional = (1-bluff)*made + bluff*eqs[0]
                w = self.params['conditional_weight']
                selected = w*conditional + (1-w)*single
                equity = max(.001, min(.999, selected)) ** (1+.68*(m-1))
                penalty = .015 + .025*max(0,len(bets)-1) + .015*(m-1)
            elif samples >= 8:
                penalty += max(-.055, min(.075, (.29-aggression)*.38))
            if call >= stack * .7:
                penalty += .025
            penalty *= self.params['bet_penalty_scale']
            value_eq = max(0., equity - penalty)
            risk = self._risk(o, call, last_bettor) if last_bettor is not None else 0.
            if value_eq >= odds + risk:
                # Strong hands raise for value, with more caution multiway.
                threshold = self.params['value_threshold'] + .045*(m-1) + .025*street
                if o['can_raise'] and value_eq > threshold and single > .84:
                    target = put + call + max(o['bb'], (pot+call) * .68)
                    if target - put > stack * .62 or spr < 1.4:
                        target = o['max_raise_to']
                    return self._raise(o, target)
                return {'action': 'call'}
            # Cheap draws already reflected in expected showdown equity;
            # implied odds are allowed only when more cards remain and deep.
            if street < 3 and bet_ratio <= .4 and spr > 2 and value_eq + .045 >= odds + risk:
                return {'action': 'call'}
            return {'action': 'fold'}
        if not o['can_raise']:
            return {'action': 'check'}
        checked = [a['seat'] for a in o['history'] if a['street'] == street and a['action'] == 'check']
        ip = all((i-o['button']-1) % len(o['stacks']) <= (h-o['button']-1) % len(o['stacks']) for i in opponents)
        last_raiser = pre[-1]['seat'] if pre else None
        pfr = last_raiser == h
        rnd = self._random()
        value = equity > self.params['value_threshold'] + .03*(m-1) - .015*street
        thin = m == 1 and equity > .59 and (street == 3 or ip)
        cbet = (pfr and street == 1 and m <= 2 and single > .39
                and rnd < self.params['cbet'] / max(1, m))
        # Bluff only heads-up, preferentially in position after a check.
        bluff = (m == 1 and (ip or checked) and .12 < single < .43
                 and rnd < (.16 if street == 3 else .12) * self.params['bluff_scale'])
        if value or thin or cbet or bluff:
            fraction = .66 if value else .42 if cbet else .50
            if value and single > .94 and spr > 1.5:
                fraction = .85
            target = put + max(o['bb'], pot * fraction)
            if target - put > stack * .68:
                target = o['max_raise_to']
            return self._raise(o, target)
        return {'action': 'check'}


PokerBot = Agent
