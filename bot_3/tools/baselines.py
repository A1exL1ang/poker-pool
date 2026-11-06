"""Simple, explicitly synthetic testing policies, not independent submissions."""
import random
from bot3.cards import card_id, hand_class


class ScriptedOpponent:
    def __init__(self, model, style="balanced", seed=1):
        self.model = model
        self.style = style
        self.seed = seed
        self.reset()

    def reset(self, table_id=None):
        self.rng = random.Random(self.seed)

    def act(self, o):
        h = next(p for p in o["players"] if p["seat"] == o["hero_seat"])
        legal = o["legal_actions"]
        call = o["to_call"]
        bb = o["big_blind"]
        pot = o["pot"]
        maxbet = o.get("current_bet", max(p["bet"] for p in o["players"]))
        hole = tuple(card_id(c) for c in o["hole_cards"])
        board = o["board"]
        passive = {"action": "check" if "check" in legal else "fold", "amount": 0}

        def raise_to(target):
            maximum = o["max_raise_to"]
            if "raise" not in legal:
                return {"action": "call", "amount": call} if "call" in legal else passive
            size = int(min(maximum, max(o["min_raise_to"], target)))
            return {"action": "raise", "amount": size}

        if self.style == "random":
            u = self.rng.random()
            if "raise" in legal and u < .12:
                return raise_to(maxbet + max(bb, pot * .7))
            if call and u < .65:
                return {"action": "call", "amount": call}
            return passive
        if not board:
            pct = self.model.percentile[hand_class(*hole)]
            width = {"tight": .17, "passive": .68, "pressure": .48, "balanced": .28}[self.style]
            if maxbet <= bb:
                if pct < width:
                    if self.style == "passive" and pct > .07:
                        return {"action": "call", "amount": call} if call else passive
                    return raise_to(bb * 3)
            eq = self.model.preflop(hole, .20)
            if pct < .045 and "raise" in legal:
                return raise_to(maxbet * 3)
            threshold = call / max(1, pot + call) + (.12 if self.style == "tight" else .06)
            if self.style == "passive":
                threshold -= .10
            if eq > threshold and (call <= h["stack"] * .25 or eq > .53):
                return {"action": "call", "amount": call} if call else passive
            return passive
        eqs, x = self.model.postflop(hole, board)
        eq = .6 * eqs[0] + .4 * eqs[4]
        opponents = sum(not p["folded"] and p["seat"] != o["hero_seat"] for p in o["players"])
        eq **= 1 + .65 * max(0, opponents - 1)
        if call:
            margin = {"tight": .13, "passive": -.07, "pressure": .025, "balanced": .06}[self.style]
            if eq > .79 and self.style != "passive":
                return raise_to(maxbet + pot)
            if eq > call / max(1, pot + call) + margin:
                return {"action": "call", "amount": call}
            return passive
        threshold = .72 if self.style == "passive" else .58
        if eq > threshold or (self.style == "pressure" and opponents == 1 and self.rng.random() < .40):
            return raise_to(h["bet"] + pot * .65)
        return passive
