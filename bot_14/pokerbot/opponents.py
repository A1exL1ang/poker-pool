"""Original diversified *test opponents*, not contest entrants or copied bots.

They deliberately span tight/value, calling, random aggression and bounded
rollout policies. Results against these synthetic fields have limited scope.
"""
from __future__ import annotations
import random
from .evaluator import evaluate
from .interface import legalize, passive

STYLES = ("tag", "nit", "station", "lag", "maniac", "rollout", "balanced")


class Opponent:
    def __init__(self, style="tag", seed=0):
        if style not in STYLES:
            raise ValueError(style)
        self.style = style
        self.rng = random.Random(seed)

    def decide(self, o):
        p = o["players"][o["hero"]]
        bb, cost = o["big_blind"], min(p["stack"], o["to_call"])
        live = [i for i, x in enumerate(o["players"]) if i != o["hero"] and not x["folded"] and x["stack"] + x["invested"] > 0]
        n = len(live)
        if not n:
            return passive(o)
        h0, h1 = o["hole_cards"]
        hi, lo = max(h0 >> 2, h1 >> 2), min(h0 >> 2, h1 >> 2)
        suited = h0 & 3 == h1 & 3
        if o["street"] == "preflop":
            e = .49 + .027 * hi if hi == lo else .28 + .021 * hi + .012 * lo + .025 * suited - .004 * max(0, hi - lo - 2)
            first = o["current_bet"] <= bb
            open_e = dict(tag=.61, nit=.68, station=.46, lag=.55, maniac=.40, rollout=.59, balanced=.60)[self.style]
            raisers = sum(x["street"] == "preflop" and x["action"] == "raise" for x in o.get("history", ()))
            if first and e >= open_e and o["can_raise"]:
                target = bb * (2.5 if self.style != "maniac" else self.rng.choice((2, 4, 8)))
                if p["stack"] < 11 * bb:
                    target = p["stack"] + p["committed"]
                if self.style == "station":
                    target = bb * 2
                return legalize(o, {"action": "raise", "amount": round(target)})
            if e > (.76 if self.style != "lag" else .70) and o["can_raise"]:
                target = max(o["min_raise_to"], 3 * o["current_bet"])
                if target > p["stack"] * .3:
                    target = p["committed"] + p["stack"]
                return legalize(o, {"action": "raise", "amount": target})
            odds = cost / max(1, o["pot"] + cost)
            if self.style == "station":
                call = e > .40 and (cost < p["stack"] * .3 or e > .58)
            elif self.style == "maniac":
                call = e > .42 or self.rng.random() < .22
            else:
                allowed = max(.49, open_e - .045 + .035 * max(0, raisers - 1))
                call = e >= allowed and e * .80 > odds + (.08 if cost > .3 * p["stack"] else .025)
            return {"action": "call"} if cost and call else passive(o)
        board = o["board"]
        score = evaluate(tuple(o["hole_cards"]) + tuple(board))
        category = score >> 20
        if self.style in ("rollout", "balanced"):
            known = set(o["hole_cards"]) | set(board)
            available = [c for c in range(52) if c not in known]
            total = 0.
            samples = 40 if self.style == "rollout" else 20
            for _ in range(samples):
                sample = self.rng.sample(available, 5 - len(board) + 2 * n)
                full = tuple(board) + tuple(sample[:5 - len(board)])
                hs = evaluate(tuple(o["hole_cards"]) + full)
                shares, best = 1, True
                offset = 5 - len(board)
                for i in range(n):
                    vs = evaluate(tuple(sample[offset + 2 * i:offset + 2 * i + 2]) + full)
                    if vs > hs:
                        best = False
                        break
                    shares += vs == hs
                total += 1 / shares if best else 0
            e = total / samples
            # An independently coded action-history discount for strong bets.
            raises = sum(x["action"] == "raise" and x["street"] != "preflop" for x in o.get("history", ()))
            e = max(.01, e - .045 * raises)
        else:
            e = (.25 + .017 * hi, .61, .81, .89, .94, .96, .985, .998, 1.)[category]
            if category == 1:
                pair_rank = (score >> 16) & 15
                high_board = max(c // 4 + 2 for c in board)
                e += .12 if pair_rank >= high_board else -.12 if pair_rank < high_board - 4 else -.03
                if hi + 2 != pair_rank and lo + 2 != pair_rank:
                    e -= .17
            elif category == 2 and len({c // 4 for c in board}) < len(board) and hi != lo:
                e -= .10
            e = max(.01, min(.999, e)) ** max(1., .73 * n)
        odds = cost / max(1, o["pot"] + cost)
        margin = dict(tag=.07, nit=.15, station=-.12, lag=.015, maniac=-.10, rollout=.015, balanced=.045)[self.style]
        if cost and e < odds + margin:
            return passive(o)
        value = dict(tag=.72, nit=.83, station=.90, lag=.63, maniac=.43, rollout=.67, balanced=.71)[self.style]
        bluff = dict(tag=.04, nit=.005, station=.0, lag=.14, maniac=.32, rollout=.08, balanced=.07)[self.style]
        if o["can_raise"] and (e > value or not cost and self.rng.random() < bluff / max(1, n)):
            target = p["committed"] + cost + round((o["pot"] + cost) * (.70 if e > .85 else .50))
            return legalize(o, {"action": "raise", "amount": target})
        return {"action": "call"} if cost else {"action": "check"}

    __call__ = decide
