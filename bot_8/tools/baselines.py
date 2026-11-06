"""Original simple reference policies for LOCAL validation only.

These are deliberately varied test opponents, not additional contest entries.
They do not use hidden cards, online services, or another team's code.
"""

from kestrel.cards import evaluate, draw_strength, hand_class
from kestrel.equity import BASE, PERCENTILE, made_quality
from kestrel.model import Action, legalize


class Baseline:
    def __init__(self, style="abc"):
        self.style = style

    def act(self, obs):
        h = next(p for p in obs.players if p.seat == obs.hero)
        n = sum(not p.folded for p in obs.players)
        call = min(obs.to_call, h.stack)
        odds = call / max(1, obs.pot + call)
        r = ((obs.hand_id * 6151 + obs.hero * 7817 + sum(obs.hole) * 6971 + obs.pot * 271) % 65521) / 65521
        pct = PERCENTILE[hand_class(*obs.hole)]
        aggressive = self.style == "lag"
        tight = self.style == "nit"
        station = self.style == "station"
        if self.style == "random":
            if r < .12 and obs.can_raise:
                return legalize(obs, Action("raise", h.committed + call + max(obs.big_blind, obs.pot // 2)))
            return Action("fold" if call and r > .8 else "call" if call else "check")
        if not obs.board:
            current = h.committed + obs.to_call
            if pct < (.055 if tight else .085) and obs.can_raise:
                target = max(round(2.8 * obs.big_blind), round(current * 3))
                if target > .4 * (h.stack + h.committed):
                    target = h.stack + h.committed
                return legalize(obs, Action("raise", target))
            raised = current > obs.big_blind
            if not raised and pct < (.12 if tight else .46 if aggressive else .29) and not station:
                return legalize(obs, Action("raise", round(2.5 * obs.big_blind)))
            cutoff = (.19 if tight else .70 if station else .48 if aggressive else .34)
            if not call:
                return Action("check")
            if pct < cutoff and odds < (.40 if aggressive else .34) and call < (.22 if tight else .40) * h.stack:
                return Action("call")
            if pct < .10 and odds < .35:
                return Action("call")
            if station and odds < .15 and call < .1 * h.stack:
                return Action("call")
            return Action("fold")
        value = evaluate(obs.hole + obs.board)
        quality = made_quality(obs.hole, obs.board, value)
        draw, _ = draw_strength(obs.hole, obs.board)
        q = min(.99, max(quality, draw + .20)) ** max(1, .67 * (n - 1))
        if obs.can_raise and (q > (.86 if tight else .76) or (aggressive and r < .22 and q > .30)):
            size = round(obs.pot * (.8 if aggressive else .65))
            target = h.committed + call + max(obs.big_blind, size)
            if target >= .6 * h.stack and q > .8:
                target = h.stack + h.committed
            return legalize(obs, Action("raise", target))
        if call:
            threshold = odds + (.19 if tight else .08 if station else .14)
            if q > threshold and (q > .50 or call < .18 * h.stack or draw > .25):
                return Action("call")
            return Action("fold")
        if not station and (q > .61 or (aggressive and r < .44)):
            return legalize(obs, Action("raise", h.committed + max(obs.big_blind, round(.45 * obs.pot))))
        return Action("check")
