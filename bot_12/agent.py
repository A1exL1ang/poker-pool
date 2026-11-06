"""bot_12: offline, range-aware no-limit Hold'em tournament policy.

The dict protocol is LOCAL and specified in README.md, not an official
contest Agent interface. Agent instances are scoped to one anonymous table.
All training assets load at import time. Decisions use no online service.
"""
from __future__ import annotations

from equity import analyze, evaluate, hand_key, preflop_equities, preflop_narrow_equities
from icm import call_threshold

DEFAULTS = {
    "open_scale": 1.15, "call_margin": .04, "value_threshold": .73,
    "bluff_scale": .6, "range_selection": 1.1, "shove_margin": .025,
    "threebet_scale": 1.0, "realization": .97,
}
_RANGES = (.05, .10, .20, .35, .50, 1.0)


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def _range_equity(es, width, narrow=None):
    width = _clamp(width, .01, 1.0)
    if width <= .05:
        if narrow is not None:
            if width <= .025:
                f = (width-.01)/.015
                return narrow[0] + f*(narrow[1]-narrow[0])
            return narrow[1] + (width-.025)/.025*(es[0]-narrow[1])
        return max(.03, es[0] - (.05 - width) * 1.1)
    for k in range(1, 6):
        if width <= _RANGES[k]:
            f = (width - _RANGES[k-1]) / (_RANGES[k] - _RANGES[k-1])
            return es[k-1] + f * (es[k] - es[k-1])
    return es[-1]


def _multiway_equity(eq, opponents, tie=None):
    """Independent-opponent split-pot approximation when tie mass is known.

    Card removal and common-board correlation mean this remains an approximation
    for multiple opponents. Its heads-up and guaranteed-chop limits are exact.
    """
    eq = _clamp(eq)
    if opponents <= 1:
        return eq
    if tie is None or tie < 1e-9:
        return eq ** (opponents ** .82)
    tie = min(tie, 2*eq, 2*(1-eq))
    if tie < 1e-9:
        return eq ** (opponents ** .82)
    win = max(0., eq-.5*tie)
    return ((win+tie)**(opponents+1)-win**(opponents+1))/((opponents+1)*tie)


def _playability(hole):
    a, b = sorted((c // 4 + 2 for c in hole), reverse=True)
    suited = (hole[0] & 3) == (hole[1] & 3)
    # An original positional ordering, distinct from heads-up raw equity.
    if a == b:
        return 36 + 4.5 * a
    gap = a - b - 1
    return 2.7*a + 1.6*b + 5.0*suited - 2.2*min(gap, 4) + (2.5 if a == 14 else 0) + (2 if a >= 11 and b >= 10 else 0) + (2 if suited and a == 14 and b <= 5 else 0)


_ORDER = []
for _a in range(13):
    for _b in range(13):
        if _a == _b:
            _h, _combos = (4*_a, 4*_b+1), 6
        elif _a > _b:
            _h, _combos = (4*_a, 4*_b), 4
        else:
            _h, _combos = (4*_b, 4*_a+1), 12
        _ORDER.append((_playability(_h), hand_key(_h), _combos))
_ORDER.sort(reverse=True)
_PERCENTILE = [0.] * 169
_count = 0
for _v, _k, _c in _ORDER:
    _PERCENTILE[_k] = (_count + .5*_c) / 1326
    _count += _c


class Agent:
    def __init__(self, seed=120012, params=None):
        self.initial_seed = int(seed)
        self.params = dict(DEFAULTS)
        if params:
            self.params.update(params)
        self.rng = int(seed) & 0xffffffff or 1
        self.stats = []
        self.last_hand = None
        self.seen = 0
        self.hand_seen = set()
        self.vpip_seen = set()
        self.pfr_seen = set()
        self.decisions = 0
        self.fallbacks = 0
        self._equity_key = None
        self._equity_value = None

    def reset(self, seed=None):
        self.__init__(self.initial_seed if seed is None else seed, self.params)

    def _random(self):
        x = self.rng
        x ^= (x << 13) & 0xffffffff
        x ^= x >> 17
        x ^= (x << 5) & 0xffffffff
        self.rng = x & 0xffffffff
        return self.rng / 4294967296.0

    def _observe(self, o):
        n = len(o["stacks"])
        if len(self.stats) != n:
            # [observed preflop opportunities, vpip, pfr, bets, calls, folds].
            self.stats = [[0, 0, 0, 0, 0, 0] for _ in range(n)]
        hand = o.get("hand_number", 0)
        if hand != self.last_hand:
            self.last_hand, self.seen = hand, 0
            self.hand_seen.clear()
            self.vpip_seen.clear()
            self.pfr_seen.clear()
            self._equity_key = None
            self._equity_value = None
        history = o.get("history", ())
        if len(history) < self.seen:
            self.seen = 0
        for event in history[self.seen:]:
            i = event["seat"]
            action = event["action"]
            st = self.stats[i]
            if event["street"] == "preflop":
                # After hero folds, later actions may never be observed. An
                # unseen decision must not count as an opponent folding.
                if action in ("fold", "check", "call", "raise") and i not in self.hand_seen:
                    st[0] += 1
                    self.hand_seen.add(i)
                if action in ("call", "raise") and i not in self.vpip_seen:
                    st[1] += 1
                    self.vpip_seen.add(i)
                if action == "raise" and i not in self.pfr_seen:
                    st[2] += 1
                    self.pfr_seen.add(i)
            if action == "raise":
                st[3] += 1
            elif action == "call":
                st[4] += 1
            elif action == "fold":
                st[5] += 1
        self.seen = len(history)

    def _profile(self, seat):
        st = self.stats[seat]
        vpip = (st[1] + 6) / (st[0] + 20)
        pfr = (st[2] + 3.6) / (st[0] + 20)
        aggression = (st[3] + 3) / (st[3] + st[4] + 7)
        return vpip, pfr, aggression

    def act(self, observation):
        self.decisions += 1
        try:
            self._observe(observation)
            if observation["street"] == "preflop":
                result = self._preflop(observation)
            else:
                result = self._postflop(observation)
            return self._legal(observation, result)
        except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, OverflowError):
            self.fallbacks += 1
            return {"action": "check" if "check" in observation.get("legal_actions", ()) else "fold"}

    def _legal(self, o, action):
        kind = action["action"]
        legal = o["legal_actions"]
        if kind == "raise" and "raise" in legal:
            lo, hi = o["min_raise_to"], o["max_raise_to"]
            target = max(lo, min(hi, int(action["amount"]))) if hi >= lo else hi
            if target > o["current_bet"]:
                return {"action": "raise", "amount": target}
        elif kind in legal:
            return {"action": kind}
        return {"action": "check" if "check" in legal else "fold"}

    def _call(self, o):
        return {"action": "check" if not o["to_call"] else "call"}

    def _raise(self, o, amount):
        if "raise" not in o["legal_actions"]:
            return self._call(o)
        hi = o["max_raise_to"]
        # Avoid creating a trivial stranded stack after an intended large bet.
        if amount > .72 * hi:
            amount = hi
        return {"action": "raise", "amount": round(amount)}

    def _eligible_pot(self, o, cost):
        cap = o["contributions"][o["hero"]] + cost
        return sum(min(cap, c) for c in o["contributions"])

    def _threshold(self, o, cost, opponents):
        pot = self._eligible_pot(o, cost)
        odds = cost / max(1, pot + cost)
        if len(opponents) == 1 and o.get("payouts") and cost >= .40 * o["stacks"][o["hero"]]:
            # A previous folded all-in/other side pot invalidates this shortcut.
            if not any(o["all_in"][i] for i in range(len(o["stacks"])) if i not in (o["hero"], opponents[0])):
                return call_threshold(o, opponents[0], cost, pot)
        return odds

    def _position(self, o):
        hero, button = o["hero"], o["button"]
        alive = [i for i, s in enumerate(o["stacks"]) if s > 0 or o["in_hand"][i]]
        order = sorted(alive, key=lambda i: ((i-button-1) % len(o["stacks"])))
        if len(alive) == 2:
            order = [button, next(i for i in alive if i != button)]
        elif order:
            order = order[2:] + order[:2]
        idx = order.index(hero)
        behind = [i for i in order[idx+1:] if o["in_hand"][i] and not o["all_in"][i]]
        postorder = sorted([i for i, b in enumerate(o["in_hand"]) if b and (i == hero or not o["all_in"][i])], key=lambda i: ((i-button-1) % len(o["stacks"])))
        ip = bool(postorder and postorder[-1] == hero)
        return behind, ip, len(alive)

    def _preflop(self, o):
        hero = o["hero"]
        hole = o["hole"]
        stack = o["stacks"][hero]
        bb = max(1, o["big_blind"])
        cost = o["to_call"]
        pot = o["pot"]
        sbets = o["street_contributions"]
        effective = stack + sbets[hero]
        opp = [i for i, live in enumerate(o["in_hand"]) if live and i != hero]
        if not opp:
            return self._call(o)
        eff = min(effective, max(o["stacks"][i] + sbets[i] for i in opp))
        depth = eff / bb
        behind, ip, alive = self._position(o)
        es = preflop_equities(hole)
        narrow = preflop_narrow_equities(hole)
        pct = _PERCENTILE[hand_key(hole)]
        raises = [e for e in o.get("history", ()) if e["street"] == "preflop" and e["action"] == "raise"]
        limpers = {e["seat"] for e in o.get("history", ()) if e["street"] == "preflop" and e["action"] == "call"}
        free = {"action": "check"} if not cost else {"action": "fold"}
        if not raises and o["current_bet"] <= bb:
            widths = (.90, .49, .44, .30, .235, .195, .17, .145)
            width = widths[min(len(behind), 7)]
            if alive <= 3 and len(behind) >= 2:
                width = .57
            if alive == 2:
                width = .82
            width *= self.params["open_scale"]
            width /= 1 + .10 * len(limpers - {hero})
            if depth <= 12 and (behind or limpers - {hero}):
                # Bounded shove EV approximation; no live rollouts.
                foldprob, eqsum, psum = 1.0, 0.0, 0.0
                targets = list(dict.fromkeys(behind + list(limpers - {hero})))
                for seat in targets:
                    vpip, pfr, ag = self._profile(seat)
                    widthcall = _clamp((.065 + 1.4 / (depth + 2)) * (vpip / .30) ** .45, .07, .75)
                    foldprob *= 1 - widthcall
                    eqsum += widthcall * _range_equity(es, widthcall, narrow)
                    psum += widthcall
                eq = eqsum / psum if psum else es[-1]
                eq *= max(.65, 1 - .07 * max(0, len(targets)-1))
                risk = min(stack, max(o["stacks"][i] + sbets[i] - sbets[hero] for i in opp))
                ev = foldprob * pot + (1-foldprob) * (eq * (pot+risk) - (1-eq)*risk)
                if ev > bb*self.params["shove_margin"] and pct < min(.90, width*1.5):
                    return self._raise(o, o["max_raise_to"])
                if depth <= 7:
                    return self._call(o) if not cost else free
            if pct < width and "raise" in o["legal_actions"]:
                size = 2.15 if depth < 25 else 2.4
                size += .9 * len(limpers - {hero})
                if not ip and len(behind) == 1 and alive > 2:
                    size += .4
                return self._raise(o, bb*size)
            # Defend a cheap blind completion with hands that realize equity.
            if cost and cost <= .5*bb and pct < .68 and len(behind) <= 1 and depth > 12:
                return self._call(o)
            if limpers and cost and cost <= bb and pct < .35 and depth > 20:
                return self._call(o)
            return free
        villain = raises[-1]["seat"] if raises else max(opp, key=lambda i: sbets[i])
        vpip, pfr, ag = self._profile(villain)
        opener = raises[0]["seat"] if raises else villain
        distance = (opener-o["button"]) % len(o["stacks"])
        late = distance == 0 or distance >= len(o["stacks"])-1 or alive <= 3
        width = (.33 if late else .17) * (pfr/.18)**.60
        if len(raises) >= 2:
            width = (.075 if len(raises) == 2 else .032 if len(raises) == 3 else .016) * (pfr/.18)**.45
        if o["current_bet"] > 5*bb and len(raises) <= 1:
            width *= max(.35, 4*bb/o["current_bet"])
        if depth <= 10:
            width = max(width, .12 + .8/(depth+2))
        eq = _range_equity(es, width, narrow)
        callers = sum(1 for i in opp if i != villain and sbets[i] >= o["current_bet"])
        eq_multi = eq ** (1 + .70*callers)
        odds = self._threshold(o, cost, opp)
        if cost >= .72*stack or o["current_bet"] >= .75*eff:
            return self._call(o) if eq_multi > odds + self.params["call_margin"] else free
        # Value 3/4-bets and small controlled suited-ace blocker component.
        canraise = "raise" in o["legal_actions"]
        premium = (pct < .027 and eq > .55) or (eq > .64 and pct < .065)
        if canraise and premium:
            if depth <= 24 or len(raises) >= 3:
                return self._raise(o, o["max_raise_to"])
            return self._raise(o, o["current_bet"]*(3.05 if ip else 3.8) + callers*o["current_bet"])
        a, b = sorted((c//4+2 for c in hole), reverse=True)
        suited = (hole[0]&3) == (hole[1]&3)
        if canraise and len(raises) == 1 and not callers and depth > 28 and suited and a == 14 and b <= 5 and self._random() < .12*self.params["threebet_scale"]:
            return self._raise(o, o["current_bet"]*(3.05 if ip else 3.8))
        realization = (.91 if ip else .81) * self.params["realization"]
        realization += .05*suited
        realization -= .025*len(behind)
        # Set mining requires a sufficiently cheap call and a deep opponent.
        if a == b and cost < .055*eff and depth > 25 and len(raises) <= 2:
            return self._call(o)
        if eq_multi*realization > odds + self.params["call_margin"] and pct < max(.15, width*1.6):
            return self._call(o)
        return free

    def _postflop(self, o):
        hero = o["hero"]
        stack = o["stacks"][hero]
        board = o["board"]
        hole = o["hole"]
        cost = o["to_call"]
        pot = o["pot"]
        bb = o["big_blind"]
        opp = [i for i, b in enumerate(o["in_hand"]) if b and i != hero]
        if not opp:
            return self._call(o)
        equity_key = (tuple(hole), tuple(board))
        if equity_key != self._equity_key:
            self._equity_value = analyze(hole, board)
            self._equity_key = equity_key
        a = self._equity_value
        history = o.get("history", ())
        pre = [e for e in history if e["street"] == "preflop" and e["action"] == "raise"]
        current = [e for e in history if e["street"] == o["street"] and e["action"] == "raise"]
        villain = current[-1]["seat"] if current else pre[-1]["seat"] if pre else opp[0]
        if villain == hero:
            villain = opp[0]
        vpip, pfr, ag = self._profile(villain)
        tightness = _clamp(.70 - 1.2*(vpip-.30), .1, .95)
        if pre and pre[-1]["seat"] != hero:
            tightness += .18
        tightness = _clamp(tightness)
        eq = a["equity_random"]*(1-tightness) + a["equity_tight"]*tightness
        tie = (a["tie_random"]*(1-tightness) + a["tie_tight"]*tightness) if a.get("tie_random") is not None else None
        behind, ip, alive = self._position(o)
        # Multiway exponent is a compact approximation, validated as such.
        multi = len(opp)
        eqmulti = _multiway_equity(eq, multi, tie)
        river = len(board) == 5
        plays_board = river and evaluate(board) == a["score"]
        if plays_board:
            high = (a["score"] >> 16) & 15
            locked = (a["category"] == 8 and high == 14)
            locked |= a["category"] == 7 and ((a["score"] >> 12) & 15) == (13 if high == 14 else 14)
            locked |= a["category"] == 4 and high == 14 and max(sum((c&3)==s for c in board) for s in range(4)) <= 2
            if locked:
                # Every remaining hand ties. Binary win/loss odds and an error
                # margin do not apply to a guaranteed, nonlosing chop.
                return self._call(o)
        if plays_board:
            # A locked royal flush on board must call for a guaranteed chop.
            if a["category"] == 8 and ((a["score"] >> 16) & 15) == 14:
                eqmulti = 1/(multi+1)
            else:
                # Preserve the tie component on paired/straight/flush boards.
                eqmulti = max(eqmulti, max(0., (2*eq-1)) / (multi+1))
        free = {"action": "check"} if not cost else {"action": "fold"}
        if cost:
            before = max(bb, pot-cost)
            size = cost / before
            value_width = _clamp(.72 - .32*min(1.5, size) + .32*(ag-.43), .18, .85)
            bluff = _clamp(.20 + .30*(ag-.43), .05, .40)
            if multi > 1:
                value_width *= .78
                bluff /= multi
            if len(current) >= 2:
                value_width *= .65
                bluff *= .6
            conditional = max(0., (eq - (1-value_width))/value_width)
            conditional = (1-bluff)*conditional + bluff*max(eq, .80)
            conditional += a["draw_strength"] * (.28 if not river else 0)
            selection = (.40 if len(board)==3 else .59 if len(board)==4 else .80)*self.params["range_selection"]
            adjusted = _clamp(eq*(1-selection) + conditional*selection)
            adjusted = _multiway_equity(adjusted, multi, tie)
            # Board locks: no raise without a private-card advantage.
            if plays_board and a["category"] == 8 and ((a["score"] >> 16)&15) == 14:
                adjusted = 1/(multi+1)
            odds = self._threshold(o, cost, opp)
            real = 1. if river or cost >= .75*stack else .95 if ip else .88
            if a["draw_strength"] > .2 and not river:
                real += .04
            margin = self.params["call_margin"]*(.5 if cost < .1*pot else 1)
            if adjusted*real < odds + margin:
                return free
            if not plays_board and "raise" in o["legal_actions"]:
                value = adjusted > .77 and eqmulti > .87
                semi = multi == 1 and not river and a["draw_strength"] > .30 and size < .8 and self._random() < .12*self.params["bluff_scale"]
                if value or semi:
                    amount = o["current_bet"] + .75*(pot+cost)
                    return self._raise(o, amount)
            return self._call(o)
        canraise = "raise" in o["legal_actions"]
        if not canraise or plays_board:
            return self._call(o)
        threshold = self.params["value_threshold"] + .035*(multi-1)
        if river:
            threshold += .035
        # Strong hands build pots; medium hands retain bluff-catching value.
        if eqmulti > threshold:
            wet = a.get("flush_cards", 0) >= 2 or a.get("straight_outs", 0) >= 1
            fraction = .70 if wet and not river else .55
            if eqmulti > .93:
                fraction = .85
            return self._raise(o, max(bb, fraction*pot))
        if multi <= 2 and not river:
            draw = a["draw_strength"]
            aggressor = bool(pre and pre[-1]["seat"] == hero)
            dry = max(sum((c&3)==s for c in board) for s in range(4)) <= 2
            probability = .0
            if draw > .19:
                probability = .42 if multi == 1 else .20
            elif aggressor and len(board)==3 and dry and multi==1 and eq>.22:
                probability = .31 if ip else .20
            elif ip and multi==1 and eq<.38:
                probability = .17 if river else .13
            if self._random() < probability*self.params["bluff_scale"]:
                return self._raise(o, max(bb, pot*(.55 if draw>.19 else .34)))
        elif river and multi == 1 and ip and eq < .32:
            # Rare blocker-biased river bluff after checks.
            blocker = max(c//4 for c in hole) >= 11
            if blocker and self._random() < .12*self.params["bluff_scale"]:
                return self._raise(o, max(bb, .7*pot))
        return self._call(o)


PokerBot = Agent
