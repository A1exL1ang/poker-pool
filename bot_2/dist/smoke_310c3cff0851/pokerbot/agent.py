"""Position/range-aware tournament Hold'em policy with a bounded decision path."""

from .cards import class_index, BITS, COUNT, evaluate
from .equity import PERCENTILE, preflop_equity, texture
from .fast_equity import equity as postflop_equity
from .icm import bubble_factor
from .model import Observation, legalize


class Agent:
    def __init__(self, seed=20261106):
        self.seed = int(seed) & 0xFFFFFFFF
        self.reset()

    def reset(self):
        self.table_id = None
        self.hand_id = None
        self.seen = 0
        self.stats = {}  # anonymous seat statistics, reset at every table
        self.hand_vpip = set()
        self.hand_pfr = set()
        self.hand_seen = set()
        self.poststats = {}
        self.risk_cache = {}

    def _observe(self, obs):
        if obs.table_id != self.table_id:
            self.reset()
            self.table_id = obs.table_id
        if obs.hand_id != self.hand_id:
            self.hand_id = obs.hand_id
            self.seen = 0
            self.hand_vpip.clear()
            self.hand_pfr.clear()
            self.hand_seen.clear()
            self.risk_cache.clear()
            for p in obs.players:
                if p.stack + p.invested > 0 and not p.sitting_out:
                    self.stats.setdefault(p.seat, [0, 0, 0])
        # The interface requires full current-hand history. Idempotent updates.
        for e in obs.history[self.seen:]:
            if e.street == 0 and e.kind in ("fold", "check", "call", "raise") and e.seat not in self.hand_seen:
                self.stats.setdefault(e.seat, [0, 0, 0])[0] += 1
                self.hand_seen.add(e.seat)
            if e.street == 0 and e.kind in ("call", "raise"):
                s = self.stats.setdefault(e.seat, [1, 0, 0])
                if e.seat not in self.hand_vpip:
                    s[1] += 1
                    self.hand_vpip.add(e.seat)
                if e.kind == "raise" and e.seat not in self.hand_pfr:
                    s[2] += 1
                    self.hand_pfr.add(e.seat)
            elif e.street > 0 and e.kind in ("fold", "check", "call", "raise"):
                s = self.poststats.setdefault(e.seat, [0, 0, 0, 0])
                s[0] += 1
                s[1] += e.kind == "raise"
                s[2] += e.kind == "call"
                s[3] += e.kind == "fold"
        self.seen = len(obs.history)

    def act(self, obs):
        """Return one legal Action. Prefer typed Observation in a timed host."""
        if isinstance(obs, dict):
            obs = Observation.from_dict(obs)
        self._observe(obs)
        hero = next(p for p in obs.players if p.seat == obs.hero)
        villains = [p for p in obs.players if p.seat != obs.hero and not p.folded
                    and not p.sitting_out and p.stack + p.invested > 0]
        if not villains:
            return legalize(obs, "check")
        if not obs.to_call and not any(p.stack > 0 for p in villains):
            return legalize(obs, "check")
        bb = obs.big_blind
        effective = min(hero.stack + hero.street_bet,
                        max(p.stack + p.street_bet for p in villains))
        # Cap inaccessible side-pot chips at hero's maximum total contribution.
        contributed = sum(p.invested for p in obs.players)
        cap = hero.invested + hero.stack
        eligible = sum(min(p.invested, cap) for p in obs.players)
        pot = max(1, obs.pot - max(0, contributed - eligible))
        # Stable per-state mixing; does not depend on decision count or timing.
        key = self.seed ^ (obs.hand_id * 0x9E3779B1) ^ (obs.hero * 0x85EBCA77)
        for c in obs.hole + obs.board:
            key = ((key ^ (c + 1)) * 0x45D9F3B) & 0xFFFFFFFF
        mix = ((key ^ (key >> 16)) & 65535) / 65536.0
        live = [p.seat for p in obs.players if not p.sitting_out and p.stack + p.invested > 0]
        circle = sorted(live)
        bpos = circle.index(obs.button) if obs.button in circle else -1
        clockwise = circle[bpos + 1:] + circle[:bpos + 1]
        if len(circle) == 2:
            pre_order = [obs.button] + [s for s in circle if s != obs.button]
        else:
            pre_order = clockwise[2:] + clockwise[:2]
        after = len(pre_order) - pre_order.index(obs.hero) - 1
        actionable = {p.seat for p in villains if p.stack > 0}
        post_order = [s for s in clockwise if s == obs.hero or s in actionable]
        in_position = bool(post_order and post_order[-1] == obs.hero)
        width, pre_raises, aggressor = self._range(obs, pre_order)
        risk = self._risk(obs, hero, villains, effective)
        if not obs.board:
            return self._preflop(obs, hero, villains, pot, effective, after,
                                 in_position, width, pre_raises, risk, mix)
        return self._postflop(obs, hero, villains, pot, effective, in_position,
                             width, aggressor, risk, mix, key)

    def _range(self, obs, order):
        raises = [e for e in obs.history if e.street == 0 and e.kind == "raise"]
        opponent_raises = [e for e in raises if e.seat != obs.hero]
        if opponent_raises:
            e = opponent_raises[-1]
            pos = order.index(e.seat) if e.seat in order else 0
            left = len(order) - pos - 1
            width = self._open_width(left, len(order))
            stats = self.stats.get(e.seat, (0, 0, 0))
            # Strong prior prevents drastic reactions to a handful of hands.
            observed_pfr = (stats[2] + 4.0) / (stats[0] + 20.0)
            expansion = min(3.0, max(.55, observed_pfr / .20))
            width *= expansion
            if len(raises) >= 3:
                width = min(width, .04 * max(1.0, expansion))
            elif len(raises) >= 2:
                width = min(width, .09 * max(1.0, expansion))
            if e.amount >= 5 * obs.big_blind and len(raises) == 1:
                width *= .72
            return max(.02, min(.85, width)), len(raises), raises[-1].seat
        calls = [e for e in obs.history if e.street == 0 and e.kind == "call" and e.seat != obs.hero]
        width = .40 if calls else 1.0
        return width, len(raises), raises[-1].seat if raises else None

    @staticmethod
    def _open_width(after, n):
        if n == 2:
            return .78
        return (.82, .59, .46, .30, .235, .19, .165, .145)[min(after, 7)]

    def _risk(self, obs, hero, villains, effective):
        if not obs.payouts or len(villains) == 1 and sum(p.stack + p.invested > 0 for p in obs.players) == 2:
            return 1.0
        live = [p for p in obs.players if p.stack + p.invested > 0 and not p.sitting_out]
        if len(live) > 8:
            return 1.0
        stacks = tuple(p.stack + p.invested for p in live)
        h = next(i for i, p in enumerate(live) if p.seat == obs.hero)
        v = max((i for i, p in enumerate(live) if p.seat != obs.hero and not p.folded),
                key=lambda i: stacks[i])
        # Initial stacks are constant throughout a hand, so this exact local
        # utility curvature can be cached across all of its betting decisions.
        wager = .65 * min(stacks[h], stacks[v])
        key = stacks, h, v, tuple(obs.payouts[:len(live)]), int(wager)
        if key not in self.risk_cache:
            self.risk_cache[key] = bubble_factor(stacks, h, v, key[3], wager)
        return self.risk_cache[key]

    def _preflop(self, obs, hero, villains, pot, eff, after, ip,
                 width, raises, risk, mix):
        a, b = obs.hole
        h = class_index(a, b)
        pct = PERCENTILE[h]
        r1, r2 = a >> 2, b >> 2
        pair = r1 == r2
        suited = (a & 3) == (b & 3)
        bb = obs.big_blind
        depth = eff / bb
        call = obs.to_call
        limps = sum(e.street == 0 and e.kind == "call" and e.seat != obs.hero for e in obs.history)
        current = max(p.street_bet for p in obs.players)
        if raises == 0 and current > bb:
            raises, width = 1, .18  # robust when a host omits optional history
        unopened = raises == 0
        # At shallow effective stacks use explicit fold-equity shove EV.
        if unopened and depth <= 12 and obs.can_raise:
            remaining = max(1, len(villains))
            caller_width = min(.46, .13 + .12 * max(0, 12 - depth) / 10)
            if len(villains) == 1:
                caller_width = max(caller_width, .30)
            fold_prob = (1 - caller_width) ** remaining
            eq = preflop_equity(a, b, caller_width)
            expected_callers = remaining * caller_width / max(.001, 1 - fold_prob)
            if expected_callers > 1:
                eq2 = preflop_equity(a, b, caller_width, opponents=2)
                eq += (eq2 - eq) * min(1.0, expected_callers - 1)
            wager = min(hero.stack, max(0, eff - hero.street_bet))
            win_pot = pot + max(0, eff - current)
            ev = fold_prob * pot + (1 - fold_prob) * (eq * win_pot - (1 - eq) * wager * risk)
            if ev > max(.10 * bb, .015 * wager) and pct < .86:
                return legalize(obs, "raise", hero.street_bet + hero.stack)
            return legalize(obs, "check")
        if unopened:
            opening = self._open_width(after, len(villains) + 1)
            opening /= 1 + max(0, risk - 1) * .28
            if limps:
                opening *= .80 if not ip else .91
            # Small suited pairs/connectors realize better than raw HU rank deep.
            adjusted = pct
            if depth >= 35 and (pair or suited and abs(r1 - r2) <= 2):
                adjusted *= .82
            if adjusted < opening:
                size = (2.2 if ip else 2.5) * bb
                if after == 1:
                    size = 3 * bb
                if limps:
                    size = (3 + limps + (not ip)) * bb
                if depth < 18 and pct < .065:
                    size = eff
                return legalize(obs, "raise", size)
            # Complete SB only at excellent odds with hands that can realize.
            if after == 1 and call <= .5 * bb and limps and (suited or pair) and pct < .60:
                return legalize(obs, "call")
            return legalize(obs, "check")
        # Opponent raises: use simulated equity versus inferred opening/3bet range.
        opponents = max(1, min(len(villains), 1 + sum(e.street == 0 and e.kind == "call" and e.amount >= current for e in obs.history)))
        eq = preflop_equity(a, b, width, opponents)
        odds = call / max(1, pot + call)
        required = odds * risk / (1 - odds + odds * risk)
        own_raised = any(e.street == 0 and e.seat == obs.hero and e.kind == "raise" for e in obs.history)
        # Avoid treating a large all-in as a routine raise/call.
        commitment = call / max(1, hero.stack)
        realization = 1.0 if commitment > .70 or depth <= 12 else .94 if ip else .84
        if opponents > 1 and commitment < .70:
            realization *= .92
        value_cut = min(.055, max(.013, width * .23))
        value = pct < value_cut and eq >= (.52 if raises == 1 else .55)
        jam = depth <= 22 or call >= hero.stack * .32 or raises >= 3
        commit_floor = min(.995, risk / (1 + risk) + .015)
        if value and obs.can_raise:
            if jam:
                if eq > max(required + .045, commit_floor):
                    return legalize(obs, "raise", obs.max_raise_to)
            else:
                target = current * (3.0 if ip else 3.8) + limps * current * .6
                if target > .40 * eff:
                    target = eff
                if target < .90 * eff or eq > commit_floor:
                    return legalize(obs, "raise", target)
        # Sparse blocker 3-bets, disabled against tight ranges and ICM pressure.
        ace_suited = suited and max(r1, r2) == 12 and min(r1, r2) in (2, 3, 4)
        if raises == 1 and depth > 35 and width > .28 and risk < 1.25 and ace_suited and mix < .25:
            return legalize(obs, "raise", current * (3 if ip else 3.7))
        implied = .0
        if pair and depth > 30 and call <= .065 * eff:
            implied = .045
        elif suited and abs(r1 - r2) <= 2 and ip and depth > 35 and call <= .045 * eff:
            implied = .025
        margin = .025 + (.025 if opponents > 1 else 0)
        if commitment > .75:
            margin = .025
        if eq * realization + implied > required + margin:
            # Calling narrow 4bet ranges with dominated hands loses large stacks.
            if raises < 3 or pair and r1 >= 9 or eq > .53 or odds < .15:
                return legalize(obs, "call")
        return legalize(obs, "check")

    def _postflop(self, obs, hero, villains, pot, eff, ip, width,
                  pre_aggressor, risk, mix, seed):
        call = obs.to_call
        n = len(villains)
        street = obs.street
        if street == 3 and evaluate(obs.board) == 0x8E0000:
            # A royal flush on the board is a guaranteed chop. An ICM risk
            # premium must never turn a risk-free, profitable call into a fold.
            return legalize(obs, "call")
        wet, state = texture(obs.board)
        events = [e for e in obs.history if e.street == street]
        facing_raises = sum(e.kind == "raise" and e.seat != obs.hero for e in events)
        pressure = 0.0
        if call:
            ratio = call / max(1, pot - call)
            pressure = min(.94, .45 + .32 * min(1.5, ratio) + .10 * max(0, n - 1))
            if facing_raises >= 2:
                pressure = .97
            bettors = [e.seat for e in reversed(events) if e.kind == "raise" and e.seat != obs.hero]
            if bettors:
                stats = self.poststats.get(bettors[0], (0, 0, 0, 0))
                if stats[0] >= 15:
                    frequency = (stats[1] + 4) / (stats[0] + 20)
                    pressure = max(.20, min(.98, pressure + .12 - .50 * frequency))
        eq, hu, score = postflop_equity(obs.hole, obs.board, width, n,
                                       pressure, state)
        cat = score >> 20
        spr = max(0, eff - hero.street_bet) / max(1, pot)
        odds = call / max(1, pot + call)
        required = odds * risk / (1 - odds + odds * risk)
        if call:
            realization = 1.0 if street == 3 or call >= hero.stack * .65 else .95 if ip else .90
            margin = .025 + (.025 if n > 1 else 0)
            if call < .12 * pot:
                margin = .005
            # Raise polarized value, avoiding thin raises on shared-board chops.
            value_threshold = .80 if n == 1 else .69
            if eq >= value_threshold and hu > .77 and (cat >= 2 or spr < 2):
                target = max(obs.min_raise_to, hero.street_bet + call + .75 * (pot + call))
                if target >= .55 * eff or spr < 1.2:
                    target = obs.max_raise_to
                commit_ok = target < .50 * eff or eq > min(.995, risk / (1 + risk) + .02)
                if commit_ok and (facing_raises < 2 or eq > .91):
                    return legalize(obs, "raise", target)
            if eq * realization > required + margin:
                return legalize(obs, "call")
            return legalize(obs, "check")
        if not any(p.stack > 0 for p in villains):
            return legalize(obs, "check")
        # Bet hands favored against the continuing range, with larger protection
        # on dynamic boards and smaller sizes for thin value.
        threshold = .63 if n == 1 else .58
        station = False
        if n == 1:
            s = self.poststats.get(villains[0].seat, (0, 0, 0, 0))
            station = s[0] >= 12 and s[2] / s[0] > .44 and s[1] / s[0] < .08
            if station:
                threshold -= .055
        if street == 3:
            threshold += .055
        if eq > threshold and (cat > 0 or hu > .83):
            fraction = .72 if wet > .5 or eq > .85 else .48
            if street == 3 and eq > .90:
                fraction = .9
            size = max(obs.big_blind, pot * fraction)
            if eff <= size * 1.6 and eq > .71:
                size = eff
            if size > .35 * hero.stack and eq < min(.995, risk / (1 + risk) + .02):
                return legalize(obs, "check")
            return legalize(obs, "raise", hero.street_bet + size)
        # Selective heads-up continuation bets and semi-bluffs. Pure air mostly
        # checks; no bluffs into all-in opponents or large multiway fields.
        initiative = pre_aggressor == obs.hero
        prior_hero_bet = any(e.seat == obs.hero and e.kind == "raise" and e.street >= 1 for e in obs.history)
        draws = self._draws(obs.hole, state, street)
        bluff = False
        if n == 1 and spr > 1.5 and risk < 1.65 and not station:
            if street == 1 and initiative:
                bluff = eq > .34 and mix < (.62 if wet < .35 else .28)
            elif street == 2 and draws and (ip or initiative or prior_hero_bet):
                bluff = eq > .23 and mix < .48
            elif street == 3 and ip and hu < .40:
                blocker = self._blocker(obs.hole, state)
                bluff = blocker and mix < .18
            elif street == 1 and ip and not events and wet < .4:
                bluff = eq > .35 and mix < .22
        if bluff:
            fraction = .34 if street == 1 and wet < .4 else .60
            return legalize(obs, "raise", hero.street_bet + max(obs.big_blind, pot * fraction))
        return legalize(obs, "check")

    @staticmethod
    def _draws(hole, state, street):
        if street == 3:
            return False
        a, b = hole
        suits = list(state[4:])
        suits[a & 3] |= BITS[a]
        suits[b & 3] |= BITS[b]
        if any(COUNT[s] == 4 for s in suits):
            return True
        mask = state[0] | BITS[a] | BITS[b]
        return any(COUNT[(mask >> k) & 31] == 4 for k in range(9))

    @staticmethod
    def _blocker(hole, state):
        for c in hole:
            if c >> 2 >= 11 and COUNT[state[4 + (c & 3)]] >= 3:
                return True
        return False
