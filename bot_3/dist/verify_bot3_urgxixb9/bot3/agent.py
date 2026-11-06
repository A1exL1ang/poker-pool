"""Independent NLHE policy over a documented local observation contract.

The official Agent interface was not supplied. See README.md for adaptation.
All actions use total street contribution (raise-to), never incremental raises.
"""
from .cards import card_id, hand_class, STRAIGHT
from .equity import EquityModel
from .icm import call_threshold


def clamp(x, low, high):
    return max(low, min(high, x))


class Agent:
    def __init__(self, seed=3, model=None, config=None):
        self.model = model if model is not None else EquityModel()
        self.config = {"open_scale": 1.0, "aggression": 1.0, "call_margin": 0.0, "hu_equilibrium": True}
        if config:
            self.config.update(config)
        self.seed = seed & 0xFFFFFFFF or 3
        self.reset()

    def reset(self, table_id=None):
        self.table_id = table_id
        self.stats = {}  # anonymized seat statistics, lifetime exactly one table
        self.last_hand = None
        self.history_cursor = 0
        self.rng = self.seed
        self.last_info = {}
        self.equity_cache_key = None
        self.equity_cache = None
        self.pf_raises = []
        self.street_raises = {s: [] for s in ("preflop", "flop", "turn", "river")}

    def _random(self):
        s = self.rng
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.rng = s & 0xFFFFFFFF
        return self.rng / 4294967296.0

    def _observe(self, obs):
        tid = obs.get("table_id")
        if tid != self.table_id:
            self.reset(tid)
        hand = obs.get("hand_id", 0)
        if hand != self.last_hand:
            self.last_hand, self.history_cursor = hand, 0
            self.pf_raises = []
            self.street_raises = {s: [] for s in ("preflop", "flop", "turn", "river")}
        history = obs.get("history", ())
        # This contract supplies current-hand history in chronological order.
        for e in history[self.history_cursor:]:
            seat = e.get("seat")
            if seat is None:
                continue
            s = self.stats.setdefault(seat, [0, 0, 0, 0, 0, 0])
            action = e.get("action", "")
            street = e.get("street")
            if action == "raise" and street in self.street_raises:
                self.street_raises[street].append(e)
                if street == "preflop":
                    self.pf_raises.append(e)
            if e.get("street") == "preflop":
                if action in ("raise", "call", "fold"):
                    s[0] += 1
                    s[1] += action in ("raise", "call")
                    s[2] += action == "raise"
            elif action in ("raise", "call", "fold", "check"):
                s[3] += 1
                s[4] += action == "raise"
                s[5] += action == "fold"
        self.history_cursor = len(history)

    def act(self, observation):
        """Return {action, amount}; no network, file reads, or sampling here.

        Validate observations at the engine boundary, outside a strict timer.
        Required fields: hole_cards, board, players, hero_seat, button, big_blind,
        pot, min_raise_to, max_raise_to, legal_actions, street, hand_id.
        """
        self._observe(observation)
        o = observation
        players = o["players"]
        hero_seat = o["hero_seat"]
        hero = next(p for p in players if p["seat"] == hero_seat)
        legal = o["legal_actions"]
        hole = tuple(card_id(c) for c in o["hole_cards"])
        board = tuple(card_id(c) for c in o["board"])
        stack = hero["stack"]
        bet = hero["bet"]
        bb = max(1, o["big_blind"])
        opp = [p for p in players if p["seat"] != hero_seat and not p.get("folded", False) and (p["stack"] > 0 or p.get("all_in", False))]
        maxbet = o.get("current_bet", max(p["bet"] for p in players))
        call = min(stack, max(0, o.get("to_call", maxbet - bet)))
        pot = max(1, o["pot"])
        maxraise = min(bet + stack, o["max_raise_to"])
        minraise = o["min_raise_to"]
        canraise = "raise" in legal and maxraise > maxbet and maxraise > bet
        # Count only chips we can win; do not chase side pots beyond our stack.
        cap = hero.get("committed", bet) + stack
        eligible_pot = sum(min(p.get("committed", p["bet"]), cap) for p in players)
        # A supplied pot can contain dead chips/antes not itemized in players.
        itemized = sum(p.get("committed", p["bet"]) for p in players)
        eligible_pot += max(0, pot - itemized)
        eligible_pot = max(1, eligible_pot)
        effective = min(stack, max((p["stack"] + max(0, p["bet"] - bet) for p in opp), default=0))
        active = [p["seat"] for p in players if p["stack"] > 0 or p.get("all_in", False)]
        n = len(active)
        order = sorted(active)
        if o["button"] in order:
            ix = order.index(o["button"])
            postorder = order[ix + 1:] + order[:ix + 1]
        else:
            postorder = order
        pending = [s for s in postorder if any(p["seat"] == s for p in opp) or s == hero_seat]
        in_position = bool(pending) and pending[-1] == hero_seat
        pf_raises = self.pf_raises
        street_raises = self.street_raises[o["street"]]
        aggressor = street_raises[-1].get("seat") if street_raises else None
        aggressor_p = next((p for p in opp if p["seat"] == aggressor), max(opp, key=lambda p: p["bet"], default=None))
        odds = call / (eligible_pot + call) if call else 0.0
        risk = 0.0
        # Exact ICM for small fields and material calls, only with actual payouts.
        payouts = o.get("payouts")
        if payouts and call and n <= 5 and len(opp) == 1 and call > stack * .15 and aggressor_p:
            live = [p for p in players if p["stack"] > 0 or p.get("all_in", False)]
            hi = next(i for i, p in enumerate(live) if p["seat"] == hero_seat)
            vi = next(i for i, p in enumerate(live) if p["seat"] == aggressor_p["seat"])
            risk = max(0.0, call_threshold([p["stack"] for p in live], payouts, hi, vi, call, eligible_pot) - odds)
        self.last_info = {"pot_odds": odds, "icm_premium": risk, "opponents": len(opp)}

        def passive():
            if "check" in legal:
                return {"action": "check", "amount": 0}
            if "fold" in legal:
                return {"action": "fold", "amount": 0}
            if "call" in legal:
                return {"action": "call", "amount": call}
            raise ValueError("no recognized legal action")

        def calling():
            return {"action": "call", "amount": call} if "call" in legal else passive()

        def raising(target):
            if not canraise:
                return calling() if call else passive()
            # Below-minimum all-in is legal only if the engine offers this range.
            low = min(minraise, maxraise)
            target = int(clamp(round(target), low, maxraise))
            if target < minraise and target != bet + stack:
                return calling() if call else passive()
            return {"action": "raise", "amount": target}

        if not opp or not stack:
            return passive()
        if not board:
            key = hand_class(*hole)
            percentile = self.model.percentile[key]
            r1, r2 = sorted((c >> 2 for c in hole), reverse=True)
            suited = (hole[0] & 3) == (hole[1] & 3)
            pair = r1 == r2
            # RFI position is number of live seats between hero and the big blind.
            if n == 2:
                left = 1 if hero_seat == o["button"] else 0
            else:
                # Preflop order is two seats after the button, then one further.
                pforder = postorder[2:] + postorder[:2]
                left = n - pforder.index(hero_seat) - 1 if hero_seat in pforder else n - 1
            if n == 2:
                width = .82 if left else .70
            elif left <= 1:
                width = .58 if left == 1 else .65
            elif left == 2:
                width = .43
            elif left == 3:
                width = .31
            elif left == 4:
                width = .25
            else:
                width = .19
            width *= self.config["open_scale"]
            depth = (stack + bet) / bb
            limpers = sum(p["bet"] >= bb and p["seat"] != hero_seat for p in opp) - 1
            limpers = max(0, limpers)
            unopened = maxbet <= bb and not pf_raises
            self.last_info.update(preflop_percentile=percentile, open_width=width)
            if unopened:
                hu_depth = min(stack + bet, opp[0]["stack"] + opp[0]["bet"]) / bb if n == 2 else 1000
                if (self.config["hu_equilibrium"] and n == 2 and hero_seat == o["button"]
                        and hu_depth <= 12 and pot <= bb * 1.5 and bet <= bb * .5 and not o.get("ante", 0)):
                    p = self.model.heads_up(hole, hu_depth, "push")
                    self.last_info.update(hu_push_probability=p, effective_bb=hu_depth)
                    return raising(maxraise) if self._random() < p else passive()
                if call == 0 and percentile > width * .80:
                    return passive()
                if depth <= 11:
                    # Short-stack push ranges; reduce marginal multiway exposure.
                    push_width = clamp(width * (1.15 if depth < 7 else .85), .12, .90) / (1 + .15 * limpers)
                    if percentile <= push_width:
                        return raising(maxraise)
                    return passive()
                if percentile <= width / (1 + .17 * limpers):
                    size = bb * ((2.15 if depth < 35 else 2.4) + .8 * limpers)
                    return raising(maxraise if effective / bb < 12 else size)
                # Complete suited/decent small blind hands against a limping field.
                if call and call <= bb * .5 and percentile < .70 and odds < .18:
                    return calling()
                return passive()
            raiser = pf_raises[-1] if pf_raises else {}
            s = self.stats.get(raiser.get("seat"), [0] * 6)
            pfr = (s[2] + 4.0) / (s[0] + 20.0)
            raise_size = maxbet / bb
            if len(pf_raises) >= 3:
                range_width = .035
            elif len(pf_raises) == 2:
                range_width = .075
            else:
                range_width = clamp(pfr * 1.5, .11, .52)
                if raise_size > 6:
                    range_width *= .55
            if maxbet >= 0.65 * (stack + bet) or raise_size > 10:
                # A shallow shove is much wider than a deep-stack overbet.
                range_width = max(range_width, .32 if raise_size < 12 and len(pf_raises) <= 1 else .07)
            eq = self.model.preflop(hole, range_width)
            callers = sum(p["bet"] == maxbet for p in opp) - 1
            if callers > 0:
                eq **= 1 + .55 * callers
            margin = self.config["call_margin"]
            self.last_info.update(equity=eq, range_width=range_width)
            allin_call = call >= stack * .85 or all(p["stack"] == 0 for p in opp)
            premium = (pair and r1 >= 10) or (r1 == 12 and r2 >= 11)
            # Value 3-bet/4-bet; blocker bluffs are infrequent and position-aware.
            value3 = eq >= (.58 if len(pf_raises) < 2 else .60) or (pair and r1 >= 10)
            bluff3 = (len(pf_raises) == 1 and in_position and suited and r1 == 12 and 1 <= r2 <= 4 and depth > 28 and self._random() < .18)
            if canraise and (value3 or bluff3) and (not allin_call or premium):
                target = maxbet * (2.8 if in_position else 3.5) + max(0, callers) * maxbet
                if effective < target * 1.7 or depth <= 18:
                    target = maxraise
                return raising(target)
            if allin_call:
                hu_depth = min(stack + bet, aggressor_p["bet"] + aggressor_p["stack"]) / bb if n == 2 and aggressor_p else 1000
                if (self.config["hu_equilibrium"] and n == 2 and len(pf_raises) == 1
                        and hero_seat != o["button"] and aggressor == o["button"] and hu_depth <= 12 and not o.get("ante", 0)):
                    p = self.model.heads_up(hole, hu_depth, "call")
                    self.last_info.update(hu_call_probability=p, effective_bb=hu_depth)
                    return calling() if self._random() < p else passive()
                return calling() if eq > odds + risk + .025 + margin else passive()
            # Realization discounts protect dominated offsuit holdings and OOP calls.
            realization = .91 if in_position else .80
            if pair or suited:
                realization += .05
            if raise_size > 5 or len(pf_raises) >= 2:
                realization -= .08
            multi_penalty = .02 * max(0, len(opp) - 2)
            if eq * realization > odds + risk + .025 + margin + multi_penalty:
                # Avoid calling large fractions of stack with weak speculative hands.
                if call <= stack * .28 or eq > .51 or premium:
                    return calling()
            if pair and call < effective * .055 and len(pf_raises) <= 1:
                return calling()
            return passive()

        cache_key = (hole, board)
        if cache_key == self.equity_cache_key:
            values, x = self.equity_cache
        else:
            values, x = self.model.postflop(hole, board)
            self.equity_cache_key = cache_key
            self.equity_cache = (values, x)
        uniform, pf50, pf20, pf8, continuing, value = values
        # Start from the preflop range, then condition on observed postflop pressure.
        if len(pf_raises) >= 2:
            base = .65 * pf8 + .35 * pf20
        elif pf_raises:
            base = .65 * pf20 + .35 * pf50
        else:
            base = .75 * pf50 + .25 * uniform
        nopp = len(opp)
        spr = max(0, effective - call) / max(1, eligible_pot + call)
        ratio = call / max(1, eligible_pot - call)
        st = self.stats.get(aggressor, [0] * 6)
        aggr = (st[4] + 3) / (st[3] + 10)
        # Bigger bets and later streets represent stronger ranges unless this seat
        # has supplied evidence of unusually frequent aggression at this table.
        pressure = clamp(.28 + .32 * ratio + .12 * (len(board) - 3) + .16 * max(0, len(street_raises) - 1) - .55 * (aggr - .30), .12, .88)
        if call:
            tight = .45 * continuing + .55 * value
            eq = (1 - pressure) * base + pressure * tight
        else:
            eq = .65 * base + .35 * continuing
        # Shared-board ties are correlated. A power law otherwise approximates
        # multiway equity; this is explicitly not an exact multiway solver.
        eq **= 1 + .72 * max(0, nopp - 1)
        board_only = len(board) == 5 and x[5:11] == x[11:17]
        if board_only:
            eq = min(eq, 1 / (nopp + 1))
        if len(board) == 5 and x[11] == 8 and x[12] == 14:
            eq = 1 / (nopp + 1)  # a royal flush on the board always chops
        elif x[5] == 8 and x[6] == 14:
            eq = .999  # exact unbeatable royal flush in hero's best five
        elif len(board) == 5 and x[11] == 7 and x[5] == 7 and x[7] == 14:
            eq = .999  # board quads with the ace kicker cannot lose
        draw = len(board) < 5 and ((x[27] == 4 and x[28] >= 1) or x[33] >= 2)
        good_draw = len(board) < 5 and ((x[27] == 4 and x[29] >= 12) or (x[27] == 4 and x[33] >= 2))
        wet = x[24] >= 2 and x[32] >= 3
        hero_was_pfr = bool(pf_raises) and pf_raises[-1].get("seat") == hero_seat
        self.last_info.update(equity=eq, raw_equities=values, spr=spr, pressure=pressure)
        if call:
            if len(board) == 5 and x[11] == 8 and x[12] == 14:
                # Every matched new chip is returned in the chop, with a share
                # of existing dead money. A binary win/lose ICM model is invalid.
                return calling()
            realization = 1.0 if len(board) == 5 or call >= stack * .80 or all(p["stack"] == 0 for p in opp) else (.94 if in_position else .87)
            # Draws have implied odds only while both players can fund a later pot.
            implied = .025 if good_draw and spr > 2 and len(board) == 3 else 0.0
            need = odds + risk + self.config["call_margin"] + (.012 if len(board) < 5 else .008)
            # Strong made hands raise for value. Low-SPR strong hands commit.
            vthreshold = .77 if nopp == 1 else .70
            strong = eq > vthreshold and (x[5] >= 2 or spr < 1.2)
            if canraise and strong:
                target = maxbet + (eligible_pot + call) * (.72 if wet else .58)
                if maxraise - target < (eligible_pot + call) * .65 or spr < 1.1:
                    target = maxraise
                return raising(target)
            if canraise and good_draw and nopp == 1 and len(street_raises) <= 1 and spr > 1.5 and self._random() < .13:
                return raising(maxbet + (eligible_pot + call) * .72)
            return calling() if eq * realization + implied >= need else passive()
        if not canraise:
            return passive()
        # Bet frequently for value against weak ranges; pot growth tracks texture.
        value_threshold = (.59 if nopp == 1 else .51) + (.035 if not in_position else 0)
        if eq > value_threshold:
            size = .65 if wet or nopp > 1 else .48
            if len(board) == 5:
                size = .70 if eq > .83 else .42
            if spr < .85 and eq > .67:
                return raising(maxraise)
            if not in_position and not hero_was_pfr and len(board) == 3 and self._random() < .55:
                return passive()  # protect checking range and allow aggressor to bet
            return raising(bet + eligible_pot * size)
        # Restrained bluffs: single opponent, positional edge or initiative, with
        # draws/blockers. Multiway pots are overwhelmingly value-driven.
        if nopp == 1 and (in_position or hero_was_pfr) and spr > .8:
            freq = 0.0
            if len(board) == 3 and hero_was_pfr and not wet:
                freq = .45 if eq > .25 else .22
            elif draw:
                freq = .36 if good_draw else .18
            elif len(board) == 5 and eq < .20 and x[1] == 14 and x[24] >= 3:
                freq = .15  # ace flush blocker, not an arbitrary river bluff
            if self._random() < freq * self.config["aggression"]:
                return raising(bet + eligible_pot * (.35 if len(board) == 3 else .65))
        return passive()

    # Explicit aliases for local embedding convenience; not official adapters.
    def get_action(self, observation):
        return self.act(observation)
