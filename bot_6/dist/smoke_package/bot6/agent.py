"""Independent range-aware, time-bounded no-limit Texas Hold'em policy."""
import random
from time import perf_counter_ns
from .cards import hand_class, CARD_MASK
from .interface import State, legalize, eligible_pots
from .equity import PERCENTILE, preflop_equity, estimate, texture, hand_quality
from .icm import risk_premium, allin_call_threshold
from .preflop import OPEN_PERCENTILE


class Agent:
    def __init__(self, seed=60062026, sample_budget=120, time_budget_us=500):
        self.rng = random.Random(seed)
        self.sample_budget = max(24, min(140, int(sample_budget)))
        self.time_budget_ns = max(0, int(time_budget_us * 1000))
        self._deadline_ns = 0
        self.table_id = None
        self.hand_id = None
        self.stats = {}
        self.last_history = ()
        self.last_seats = ()
        self.last_equity = None
        self.last_samples = None

    def reset(self, table_id=None):
        self.table_id = table_id
        self.hand_id = None
        self.stats.clear()
        self.last_history = ()
        self.last_seats = ()

    def act(self, observation):
        self._deadline_ns = perf_counter_ns() + self.time_budget_ns if self.time_budget_ns else 0
        state = observation if isinstance(observation, State) else State.from_dict(observation)
        self._observe(state)
        if state.me.stack <= 0:
            return legalize(state, "check")
        if not state.opponents:
            return legalize(state, "check")
        if state.board:
            return self._postflop(state)
        return self._preflop(state)

    # A common adapter name, with exactly the same LOCAL schema.
    get_action = act

    def _observe(self, state):
        if self.table_id != state.table_id:
            self.reset(state.table_id)
        if self.hand_id != state.hand_id:
            if self.hand_id is not None:
                vpip, pfr, observed = set(), set(), set()
                for h in self.last_history:
                    if h.get("street") == "preflop":
                        if h.get("action") in ("fold", "check", "call", "raise"):
                            observed.add(h.get("seat"))
                        if h.get("action") in ("call", "raise"):
                            vpip.add(h.get("seat"))
                        if h.get("action") == "raise":
                            pfr.add(h.get("seat"))
                for seat in self.last_seats:
                    if seat not in observed:
                        continue
                    count, voluntary, raises = self.stats.get(seat, (0, 0, 0))
                    self.stats[seat] = (count + 1, voluntary + (seat in vpip), raises + (seat in pfr))
            self.hand_id = state.hand_id
        self.last_history = state.history
        self.last_seats = tuple(p.seat for p in state.players if p.seat != state.hero)

    @staticmethod
    def _position(state, seat=None):
        seat = state.hero if seat is None else seat
        seats = sorted(p.seat for p in state.players)
        # If a local simulator uses a dead button, clockwise ordering still
        # works; it should normally supply a participating button seat.
        after = [x for x in seats if x > state.button] + [x for x in seats if x <= state.button]
        if len(seats) == 2:
            return .68 if seat == state.button else .50
        sb, bb = after[:2]
        if seat == sb:
            return .47
        if seat == bb:
            return .42
        order = after[2:]
        remaining = len(order) - order.index(seat) - 1
        return (.48, .34, .26, .21, .18, .155)[min(5, remaining)]

    @staticmethod
    def _in_position(state):
        active = sorted(p.seat for p in state.players if not p.folded)
        order = [x for x in active if x > state.button] + [x for x in active if x <= state.button]
        return bool(order) and order[-1] == state.hero

    def _ranges(self, state):
        widths = {p.seat: 1. for p in state.opponents}
        pressures = {p.seat: 0 for p in state.opponents}
        raises = 0
        aggressor = None
        for h in state.history:
            seat, action, street = h.get("seat"), h.get("action"), h.get("street")
            if street == "preflop":
                if action == "raise":
                    raises += 1
                    aggressor = seat
                    if seat in widths:
                        width = self._position(state, seat) if raises == 1 else .085 if raises == 2 else .035
                        villain = next(p for p in state.opponents if p.seat == seat)
                        if villain.all_in:
                            jam_bb = (villain.contributed + villain.stack) / state.big_blind
                            if jam_bb <= 8:
                                width = max(width, .42 if raises == 1 else .20)
                            elif jam_bb <= 15:
                                width = max(width, .26 if raises == 1 else .13)
                        count, vpip, pfr = self.stats.get(seat, (0, 0, 0))
                        if count >= 8:
                            rate = (pfr + 3.2) / (count + 20)
                            width *= max(.65, min(1.7, rate / .16))
                        widths[seat] = min(.90, width)
                elif seat in widths and action == "call":
                    width = .65 if raises == 0 else .28 if raises == 1 else .10 if raises == 2 else .045
                    count, vpip, pfr = self.stats.get(seat, (0, 0, 0))
                    if count >= 8:
                        width *= max(.65, min(1.8, (vpip + 6) / (count + 20) / .30))
                    widths[seat] = min(widths[seat], width)
            elif seat in widths:
                # Earlier-street aggression narrows the preflop mixture a
                # little; current-street action uses board-aware likelihood.
                if street == state.street:
                    if action == "raise":
                        pressures[seat] = max(1, pressures[seat] + 1)
                    elif action == "call" and pressures[seat] == 0:
                        pressures[seat] = -1
                elif action == "raise":
                    widths[seat] = max(.045, widths[seat] * .82)
        return [(widths[p.seat], pressures[p.seat]) for p in state.opponents], raises, aggressor

    def _preflop(self, s):
        me, bb, call = s.me, s.big_blind, s.to_call
        opponents = s.opponents
        effective = min(me.stack + me.bet, max(v.stack + v.bet for v in opponents))
        depth = effective / bb
        name = hand_class(*s.hole)
        rank1, rank2 = sorted((c // 4 + 2 for c in s.hole), reverse=True)
        pair, suited = rank1 == rank2, s.hole[0] % 4 == s.hole[1] % 4
        percentile = PERCENTILE[name]
        playable = percentile - (.035 if suited else 0) - (.025 if suited and rank1 - rank2 <= 2 else 0)
        width = self._position(s)
        ip = self._in_position(s)
        specs, raises, aggressor = self._ranges(s)
        highest = max(p.bet for p in s.players)
        # Observations without history still recognize a live raise.
        if highest > bb and raises == 0:
            raises = 1
            specs = [(min(w, .20), p) for w, p in specs]
        limpers = sum(h.get("street") == "preflop" and h.get("action") == "call" for h in s.history)
        risk = risk_premium(s)
        roll = self.rng.random()
        if raises == 0:
            if depth <= 12:
                # Push/fold at low SPR; widen as blinds dominate the stack.
                jam_width = width * (.92 if depth > 8 else 1.16 if depth > 5 else 1.55)
                jam_width = min(.90, jam_width - risk * 1.4 - .04 * min(limpers, 3))
                if playable < jam_width:
                    return legalize(s, "raise", s.max_raise_to)
                return legalize(s, "check" if not call else "fold")
            open_width = width - .045 * min(limpers, 3) - .7 * risk
            if OPEN_PERCENTILE[name] < open_width:
                target = (2.2 if depth < 45 else 2.5) * bb + limpers * bb
                if not ip and limpers:
                    target += .5 * bb
                if target - me.bet >= .32 * effective:
                    target = s.max_raise_to
                return legalize(s, "raise", target)
            if not call:
                return legalize(s, "check")
            # Cheap suited/pocket-pair overlimps when there are implied odds.
            if limpers >= 2 and call <= bb and depth >= 25 and (pair or suited and rank1-rank2 <= 3):
                return legalize(s, "call")
            return legalize(s, "fold")

        main_width = min((w for w, p in specs), default=.2)
        eq = preflop_equity(s.hole, main_width)
        # Approximate multiway adjustment; the postflop sampler is joint.
        involved = sum(v.bet >= min(highest, me.bet + call) or v.all_in for v in opponents)
        eq_multi = eq ** (1. + .65 * max(0, involved - 1))
        contestable, _layers = eligible_pots(s)
        odds = call / max(1, contestable)
        self.last_equity, self.last_samples = eq_multi, 20000
        is_allin_call = call >= me.stack * .96 or all(v.all_in for v in opponents)
        if is_allin_call:
            # Call ranges tighten for deep jams; blind steals stay wider.
            exact = allin_call_threshold(s, contestable)
            threshold = exact if exact is not None else odds + risk
            if eq_multi > threshold + .012:
                return legalize(s, "call")
            return legalize(s, "fold")
        value = eq > (.58 if raises == 1 else .60 if raises == 2 else .64)
        if pair and rank1 >= 13:
            value = True
        if value:
            target = highest * (3.0 if ip else 3.7)
            target += max(0, involved - 1) * highest
            if depth <= 24 or target - me.bet >= effective * .35:
                target = s.max_raise_to
            return legalize(s, "raise", target)
        # Occasional blocker 3-bets, reduced against tight opponents and risk.
        blocker = suited and rank1 == 14 and rank2 <= 5
        if raises == 1 and blocker and ip and depth >= 35 and main_width >= .25 and roll < .17 * (1 - 4 * risk):
            return legalize(s, "raise", highest * 3.0)
        if not call:
            return legalize(s, "check")
        realization = (.93 if ip else .83) + (.05 if suited else 0) + (.06 if pair else 0)
        if involved > 1:
            realization -= .07
        # Large fractions of the stack have fewer future realization losses.
        realization = min(1., realization + .12 * min(1., call / max(1, effective) * 4))
        margin = .035 if raises <= 1 else .055
        if eq_multi * realization > odds + risk + margin:
            # Avoid defending weak offsuit hands just because nominal pot
            # odds look attractive: domination worsens future realization.
            if call <= .14 * effective or eq_multi > .47:
                return legalize(s, "call")
        if pair and depth >= 30 and call <= effective * .055 and call <= s.pot * .36:
            return legalize(s, "call")
        return legalize(s, "fold")

    def _postflop(self, s):
        me, opponents = s.me, s.opponents
        specs, _raises, pre_aggressor = self._ranges(s)
        contestable, layers = eligible_pots(s)
        n = len(opponents)
        call = s.to_call
        pot = max(s.big_blind, s.pot)
        ip = self._in_position(s)
        risk = risk_premium(s)
        rank_mask, high, wet = texture(s.board)
        bm = sum(CARD_MASK[c] for c in s.board)
        made, draw = hand_quality(*s.hole, bm, rank_mask, high, len(s.board))
        remaining = min(me.stack, max(v.stack + max(0, v.bet - me.bet) for v in opponents))
        spr = remaining / max(1, pot + call)
        river = len(s.board) == 5
        roll = self.rng.random()
        allin = call >= me.stack * .96 or all(v.all_in for v in opponents)
        exact = allin_call_threshold(s, contestable) if allin and call else None
        eq, ess = estimate(s.hole, s.board, specs, self.rng, self.sample_budget, layers, self._deadline_ns)
        self.last_equity, self.last_samples = eq, ess
        uncertainty = .012 + .03 * max(0., 1 - ess / 28)
        # Marginal calls require a little more evidence than value raises.
        if call:
            odds = call / max(1, contestable)
            realization = 1. if river or allin else min(1., (.91 if ip else .83) + draw * .25 + (.08 if spr < 2 else 0))
            effective_eq = eq * realization
            threshold = odds + risk * min(1., call / max(1, me.stack) * 2.5) + uncertainty
            if allin:
                if exact is not None:
                    threshold = exact + uncertainty
            # A bare board-only hand must not pay substantial bets merely
            # because everyone can play the board; the equity sampler counts
            # fractional ties correctly and pot odds handle cheap chops.
            if effective_eq < threshold:
                return legalize(s, "fold")
            raise_threshold = (.79 if river else .73) + .035 * (n - 1) + risk
            if eq > min(.95, raise_threshold) and "raise" in s.legal:
                target = me.bet + call + (pot + call) * (.75 if wet > .5 else .65)
                if target - me.bet > .55 * remaining or spr < 1.35:
                    target = s.max_raise_to
                # Keep a small trapping component when strong and dry.
                if roll > (.08 if n == 1 and wet < .4 else 0):
                    return legalize(s, "raise", target)
            if not river and n == 1 and draw >= .20 and eq > .43 and spr > 2 and roll < .09:
                return legalize(s, "raise", me.bet + call + (pot + call) * .8)
            return legalize(s, "call")

        # Betting into players with no chips cannot produce fold equity.
        if not any(v.stack > 0 for v in opponents):
            return legalize(s, "check")
        if river:
            value_threshold = .69 if n == 1 else .80 if n == 2 else .88
        else:
            value_threshold = (.57 if ip else .61) if n == 1 else .68 if n == 2 else .76
        value_threshold += risk * .4
        if eq > value_threshold:
            fraction = .75 if wet >= .64 or n > 1 else .55 if eq > .82 else .38
            if river and eq > .90:
                fraction = .85
            target = me.bet + pot * fraction
            if target - me.bet >= .57 * remaining or spr < .85:
                target = s.max_raise_to
            # Check some marginal showdown hands out of position.
            if not ip and eq < value_threshold + .09 and roll < .25:
                return legalize(s, "check")
            return legalize(s, "raise", target)

        if n == 1 and spr > 1.0:
            # Selective semi-bluffs and dry-board continuation bets. Reduce
            # bluffing against high-VPIP opponents inferred at this table.
            villain = opponents[0].seat
            count, vpip, pfr = self.stats.get(villain, (0, 0, 0))
            station = (vpip + 6) / (count + 20)
            scale = max(.35, min(1.3, .34 / station)) * (1 - 3 * risk)
            cbet = pre_aggressor == s.hero and len(s.board) == 3
            if draw >= .11 and eq > .24:
                frequency = (.52 if ip else .32) * scale
            elif cbet and wet < .58:
                frequency = (.40 if ip else .23) * scale
            elif river and made < .25 and ip:
                # Bluff candidates with a high card that can block strong
                # calls; low showdown value avoids wasting medium hands.
                frequency = (.13 if max(c // 4 for c in s.hole) >= 11 else .035) * scale
            else:
                frequency = .06 * scale if ip and eq < .35 else 0.
            if roll < frequency:
                return legalize(s, "raise", me.bet + pot * (.66 if river else .40 if cbet else .60))
        return legalize(s, "check")
