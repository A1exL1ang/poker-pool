"""Independent tournament Hold'em policy with no gameplay network calls."""
import random

from .cards import evaluate, hand_index, STRAIGHT
from .equity import Equity, PERCENTILE, preflop_equity
from .icm import risk_premium
from .protocol import legalize, normalize, safe_action


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


def opening_score(hole):
    """Equity ordering adjusted for playability, not a claim of GTO charts."""
    a, b = sorted((c >> 2 for c in hole), reverse=True)
    suited = hole[0] & 3 == hole[1] & 3
    value = PERCENTILE[hand_index(*hole)]
    if a == b:
        value -= 0.30 if a <= 6 else 0.06
    elif suited:
        gap = a - b
        if gap == 1:
            value -= 0.30
        elif gap == 2:
            value -= 0.18
        elif gap == 3:
            value -= 0.08
        if a == 12 and b <= 3:
            value -= 0.06
    elif a == 12 and b < 7:
        value += 0.08
    return clamp(value, 0.0, 1.0)


def board_features(hole, board):
    ranks = [c >> 2 for c in board]
    high = max(ranks)
    suits = [0, 0, 0, 0]
    mask = 0
    for c in hole + board:
        suits[c & 3] += 1
        mask |= 1 << (c >> 2)
    flush_draw = len(board) < 5 and any(suits[c & 3] == 4 for c in hole)
    nut_flush_draw = flush_draw and any(c >> 2 == 12 and suits[c & 3] == 4 for c in hole)
    straight_outs = 0
    if len(board) < 5 and not STRAIGHT[mask]:
        for r in range(13):
            if not mask & (1 << r) and STRAIGHT[mask | (1 << r)]:
                straight_outs += 1
    strong_draw = nut_flush_draw or flush_draw and straight_outs > 0 or straight_outs >= 2
    top_pair = any(c >> 2 == high for c in hole)
    overpair = hole[0] >> 2 == hole[1] >> 2 and hole[0] >> 2 > high
    board_suits = [sum(c & 3 == s for c in board) for s in range(4)]
    wet = max(board_suits) >= 2 and len(board) == 3 or max(board_suits) >= 3
    unique = sorted(set(ranks))
    connected = any(unique[i + 2] - unique[i] <= 4 for i in range(len(unique) - 2))
    return {
        "category": evaluate(hole + board) >> 20,
        "flush_draw": flush_draw,
        "strong_draw": strong_draw,
        "top_pair": top_pair,
        "overpair": overpair,
        "wet": wet or connected,
        "paired": len(set(ranks)) < len(ranks),
        "four_flush": max(board_suits) >= 4,
        "ace_blocker": any(c >> 2 == 12 for c in hole),
    }


class Agent:
    """Call act(observation) -> {'action': ..., 'amount': ...}.

    Construction loads immutable local assets. reset() clears all table state.
    Persistent identity is only a seat number within the current table.
    """
    def __init__(self, seed=5, equity=None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.equity = equity if equity is not None else Equity()
        self.table_id = None
        self.hand_id = None
        self.stats = {}
        self.history_seen = 0
        self.vpip_seen = set()
        self.pfr_seen = set()
        self.risk = 0.0
        self.last_error = None
        self.last_info = {}

    def reset(self, table_id=None):
        self.rng.seed(self.seed)
        self.table_id = table_id
        self.hand_id = None
        self.stats.clear()
        self.equity.reset()
        self.history_seen = 0
        self.vpip_seen.clear()
        self.pfr_seen.clear()
        self.risk = 0.0
        self.last_error = None
        self.last_info = {}

    def _update(self, obs, players, hero):
        table_id = obs.get("table_id", "local")
        if table_id != self.table_id:
            self.reset(table_id)
        hand_id = obs.get("hand_id", 0)
        if hand_id != self.hand_id:
            self.hand_id = hand_id
            self.equity.reset()
            self.history_seen = 0
            self.vpip_seen.clear()
            self.pfr_seen.clear()
            for i, p in enumerate(players):
                if p["stack"] + p.get("committed", 0) > 0:
                    self.stats.setdefault(i, [0, 0, 0, 0, 0])[0] += 1
            self.risk = 0.0
            payouts = obs.get("payouts")
            if payouts:
                remaining = [i for i, p in enumerate(players) if p["stack"] + p.get("committed", 0) > 0]
                stacks = [players[i]["stack"] + players[i].get("committed", 0) for i in remaining]
                h = remaining.index(hero)
                v = max((j for j in range(len(stacks)) if j != h), key=lambda j: stacks[j], default=h)
                self.risk = risk_premium(stacks, h, v, payouts)
        history = obs.get("history", [])
        # History is the full, append-only prefix for this hand. A shorter
        # prefix is treated as an adapter inconsistency, not replayed twice.
        for event in history[self.history_seen:]:
            seat = event.get("player")
            if seat not in self.stats:
                continue
            stat = self.stats[seat]
            action = event.get("action")
            if event.get("street") == "preflop":
                if action in ("call", "raise") and seat not in self.vpip_seen:
                    stat[1] += 1
                    self.vpip_seen.add(seat)
                if action == "raise" and seat not in self.pfr_seen:
                    stat[2] += 1
                    self.pfr_seen.add(seat)
            else:
                if action == "raise":
                    stat[3] += 1
                if action in ("call", "check", "fold"):
                    stat[4] += 1
        self.history_seen = max(self.history_seen, len(history))

    def act(self, observation):
        try:
            return self._act(observation)
        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_info = {"fallback": self.last_error}
            return safe_action(observation)

    def _act(self, obs):
        hole, board, players, hero, legal = normalize(obs)
        self.last_error = None
        self._update(obs, players, hero)
        me = players[hero]
        stack, bet = me["stack"], me.get("bet", 0)
        call = legal.get("call", 0)
        pot = max(obs["pot"], 1)
        # Exclude side-pot money that a short-stack caller cannot win.
        if call and all("committed" in p for p in players):
            ceiling = me.get("committed", 0) + call
            accounted = sum(p["committed"] for p in players)
            pot = max(1, sum(min(p["committed"], ceiling) for p in players) + max(0, obs["pot"] - accounted))
        bb = obs["big_blind"]
        active = [i for i, p in enumerate(players) if i != hero and not p.get("folded", False) and p["stack"] + p.get("committed", p.get("bet", 0)) > 0]
        if not active:
            return legalize("call", 0, legal)
        effective = min(stack, max(players[i]["stack"] + max(0, players[i].get("bet", 0) - bet) for i in active))
        button = int(obs.get("button", 0))
        nseats = len(players)
        order = sorted(active + [hero], key=lambda i: (i - button - 1) % nseats)
        in_position = order[-1] == hero
        history = obs.get("history", [])
        pre = [e for e in history if e.get("street") == "preflop"]
        raises = [e for e in pre if e.get("action") == "raise"]
        aggr = raises[-1]["player"] if raises else None
        width = self._range_width(players, active, raises, pre, obs)
        if not board:
            action, amount, info = self._preflop(hole, players, hero, active, pot, call, stack, bet, effective, bb, button, raises, pre, width, in_position, legal)
        else:
            action, amount, info = self._postflop(hole, board, players, hero, active, pot, call, stack, bet, effective, bb, history, width, in_position, aggr, legal)
        info.update({"range_width": round(width, 3), "risk_premium": round(self.risk, 3)})
        self.last_info = info
        return legalize(action, amount, legal)

    def _range_width(self, players, active, raises, pre, obs):
        if not raises:
            return 0.65 if any(e.get("action") == "call" for e in pre) else 1.0
        opponent_raises = [e for e in raises if e.get("player") in active]
        if not opponent_raises:
            widths = []
            for seat in active:
                st = self.stats.get(seat)
                if st and st[0] >= 12:
                    vpip = (st[1] + 7.0) / (st[0] + 20.0)
                    widths.append(clamp(vpip * 1.15, 0.18, 0.85))
            return min(widths) if widths else 0.38 if len(active) == 1 else 0.30
        last = opponent_raises[-1]
        seat = last["player"]
        nraises = len(raises)
        width = 0.22 if nraises == 1 else 0.08 if nraises == 2 else 0.035
        if nraises == 1:
            pos = (seat - obs.get("button", 0)) % len(players)
            if pos in (0, 1):
                width = 0.40
            elif pos == len(players) - 1:
                width = 0.30
            if last.get("amount", 0) >= obs["big_blind"] * 7:
                width *= 0.65
        st = self.stats.get(seat)
        if st and st[0] >= 8:
            observed = (st[2] + 4.0) / (st[0] + 20.0)
            width *= clamp(observed / 0.20, 0.60, 1.7)
        return clamp(width, 0.025, 0.85)

    def _preflop(self, hole, players, hero, active, pot, call, stack, bet, effective, bb, button, raises, history, width, ip, legal):
        ranks = sorted((c >> 2 for c in hole), reverse=True)
        hi, lo = ranks
        pair = hi == lo
        suited = hole[0] & 3 == hole[1] & 3
        score = opening_score(hole)
        alive = [i for i, p in enumerate(players) if p["stack"] + p.get("committed", 0) > 0]
        # Heads-up button posts the small blind; otherwise action starts after BB.
        if len(alive) == 2:
            pre_order = [button, next(i for i in alive if i != button)]
        else:
            cyclic = sorted(alive, key=lambda i: (i - button - 1) % len(players))
            pre_order = cyclic[2:] + cyclic[:2]
        behind = sum(i in active and players[i]["stack"] > 0 for i in pre_order[pre_order.index(hero) + 1:])
        thresholds = (0.64, 0.59, 0.45, 0.30, 0.23, 0.19, 0.16, 0.14)
        threshold = thresholds[min(behind, 7)]
        if len(alive) == 2 and behind == 1:
            threshold = 0.78
        limpers = sum(e.get("action") == "call" for e in history)
        unopened = not raises and max((p.get("bet", 0) for p in players), default=0) <= bb
        eq = preflop_equity(hole, width)
        info = {"street": "preflop", "equity": round(eq, 3), "opening_score": round(score, 3), "behind": behind}
        risk = self.risk * min(1.0, call / max(stack, 1) * 2)
        odds = call / (pot + call) if call else 0
        short = effective / bb <= 12
        premium = pair and hi >= 10 or hi == 12 and lo == 11
        if unopened:
            if short and "raise" in legal:
                # Approximate one-caller push EV, with a multi-caller discount.
                caller_width = clamp(0.11 + 0.8 / (effective / bb + 4), 0.13, 0.35)
                equity_called = preflop_equity(hole, caller_width)
                n = max(behind, limpers, 1)
                folds = (1 - caller_width) ** n * (0.65 ** limpers)
                if n > 3:
                    equity_called *= 0.94
                wager = min(stack, max(effective, call))
                ev = folds * pot + (1 - folds) * (equity_called * (pot + wager) - (1 - equity_called) * wager)
                if ev > (0.04 + self.risk) * bb and score < 0.84:
                    info["reason"] = "short_stack_push"
                    return "raise", bet + stack, info
            if score < threshold * (0.91 ** min(limpers, 3)):
                size = (2.25 + min(limpers, 4)) * bb
                if not ip:
                    size += 0.35 * bb
                if size - bet > 0.38 * stack:
                    size = bet + stack
                info["reason"] = "open_or_isolate"
                return "raise", size, info
            if call and effective >= 18 * call and (pair or suited and hi - lo <= 2) and limpers:
                info["reason"] = "cheap_speculative_call"
                return "call", call, info
            return "fold", 0, info
        nr = len(raises)
        raiser_allin = any(players[i]["stack"] == 0 and players[i].get("bet", 0) > bet for i in active)
        if call >= 0.45 * stack or raiser_allin and call >= 4 * bb:
            eq_multi = eq ** (1 + 0.70 * max(0, len(active) - 1))
            if eq_multi >= odds + risk + 0.025:
                return "call", call, info
            return "fold", 0, info
        value_raise = premium if nr <= 1 else pair and hi >= 11
        if nr <= 1 and width >= 0.30 and (pair and hi >= 8 or hi == 12 and lo >= 10 and suited):
            value_raise = True
        if value_raise and "raise" in legal:
            target = max(p.get("bet", 0) for p in players) * (3.0 if ip else 3.7)
            if effective <= 25 * bb or target - bet > 0.36 * effective:
                target = bet + stack
            return "raise", target, info
        # Low-frequency blocker three-bets versus a wide late-position opener.
        if nr == 1 and width >= 0.28 and suited and hi == 12 and lo <= 3 and effective >= 30 * bb and len(active) <= 3 and self.rng.random() < 0.18:
            return "raise", max(p.get("bet", 0) for p in players) * (3.2 if ip else 3.8), info
        realization = (0.92 if ip else 0.78)
        if suited:
            realization += 0.05
        eq_multi = eq ** (1 + 0.72 * max(0, len(active) - 1))
        # Opponents who have not responded to the raise are not all callers.
        matched = sum(players[i].get("bet", 0) >= bet + call for i in active)
        if len(active) > matched + 1:
            eq_multi = eq ** (1 + 0.68 * max(0, matched - 1) + 0.08 * (len(active) - matched - 1))
        if eq_multi * realization >= odds + risk + 0.025:
            return "call", call, info
        if pair and call <= effective / 18 and nr <= 2:
            return "call", call, info
        return "fold", 0, info

    def _postflop(self, hole, board, players, hero, active, pot, call, stack, bet, effective, bb, history, width, ip, pre_aggressor, legal):
        street = ("flop", "turn", "river")[len(board) - 3]
        # These public boards are unbeatable for EVERY private hand. Betting
        # cannot lose chips at showdown, and folding surrenders a guaranteed
        # share. Handle them exactly instead of powering heads-up tie equity.
        if len(board) == 5:
            board_score = evaluate(board)
            cat = board_score >> 20
            high = (board_score >> 16) & 15
            kicker = (board_score >> 12) & 15
            suits = [sum(c & 3 == s for c in board) for s in range(4)]
            locked = cat == 8 and high == 13 or cat == 4 and high == 13 and max(suits) <= 2 or cat == 7 and (kicker == 13 or high == 13 and kicker == 12)
            if locked:
                return ("raise" if "raise" in legal else "call"), bet + stack, {"street": street, "equity": 1 / (len(active) + 1), "reason": "unbeatable_shared_board"}
        events = [e for e in history if e.get("street") == street]
        bets = [e for e in events if e.get("action") == "raise"]
        ratio = call / max(1, pot - call)
        pressure = 0 if not call else 1 if ratio <= 0.45 else 2 if ratio <= 1.0 else 3
        if len(bets) >= 2:
            pressure = 3
        if call and bets:
            bettor = self.stats.get(bets[-1].get("player"))
            if bettor and bettor[3] + bettor[4] >= 20:
                aggression = (bettor[3] + 5) / (bettor[3] + bettor[4] + 20)
                if aggression < 0.16:
                    pressure = min(3, pressure + 1)
                elif aggression > 0.48 and len(bets) < 2:
                    pressure = max(1, pressure - 1)
        f = board_features(hole, board)
        eq = self.equity.estimate(hole, board, width, pressure)
        n = len(active)
        equity = eq ** (1 + 0.80 * (n - 1))
        # The power approximation underestimates shared-board draw covariance.
        if n > 1 and f["strong_draw"]:
            equity += min(0.08, (n - 1) * 0.025)
        odds = call / (pot + call) if call else 0
        spr = effective / pot
        info = {"street": street, "equity": round(equity, 3), "heads_up_equity": round(eq, 3), "pot_odds": round(odds, 3), "pressure": pressure}
        risk = self.risk * min(1.0, (call if call else pot * 0.5) / max(stack, 1) * 2)
        if call:
            margin = 0.015 if len(board) == 5 else 0.025 if ip else 0.055
            if f["strong_draw"] and len(board) < 5 and effective >= call * 4:
                margin -= 0.025
            if equity < odds + risk + margin:
                info["reason"] = "insufficient_call_equity"
                return "fold", 0, info
            raise_threshold = 0.80 if n == 1 else 0.86
            if spr < 1.4:
                raise_threshold -= 0.09
            # Raise value hands; protect medium-strength hands by calling.
            if equity > raise_threshold + risk and "raise" in legal:
                target = bet + call + (pot + call) * (0.75 if f["wet"] else 0.60)
                if target - bet >= 0.50 * effective:
                    target = bet + stack
                return "raise", target, info
            if n == 1 and f["strong_draw"] and len(board) < 5 and pressure <= 1 and spr > 2.5 and self.rng.random() < 0.16 and "raise" in legal:
                return "raise", bet + call + 0.75 * (pot + call), info
            return "call", call, info
        if "raise" not in legal:
            return "check", 0, info
        # Value thresholds tighten multiway, on later streets, and when OOP.
        value_threshold = (0.63 if len(board) == 3 else 0.66 if len(board) == 4 else 0.69) + (0.04 if not ip else 0) + min(0.08, 0.025 * (n - 1))
        if equity >= value_threshold + risk:
            fraction = 0.68 if f["wet"] else 0.46
            if equity >= 0.88:
                fraction = 0.85 if spr >= 1.8 else 0.75
            target = bet + max(bb, fraction * pot)
            if target - bet > 0.60 * effective and equity > 0.78:
                target = bet + stack
            info["reason"] = "value_bet"
            return "raise", target, info
        # Equity-backed continuation bets and occasional river blocker bluffs.
        cbet = pre_aggressor == hero and not bets
        checked_to = sum(e.get("action") == "check" for e in events) >= max(1, n)
        frequency = 0.0
        if n == 1 and len(board) < 5:
            if f["strong_draw"]:
                frequency = 0.65 if ip else 0.35
            elif cbet and not f["wet"] and equity >= 0.30:
                frequency = 0.48 if ip else 0.28
            elif ip and checked_to and equity >= 0.35:
                frequency = 0.24
        elif n == 1 and len(board) == 5 and ip and checked_to and equity < 0.30 and f["ace_blocker"]:
            frequency = 0.15
        if frequency and self.rng.random() < frequency and spr > 1.2:
            info["reason"] = "selective_bluff"
            return "raise", bet + max(bb, pot * (0.34 if cbet and not f["wet"] else 0.60)), info
        return "check", 0, info

