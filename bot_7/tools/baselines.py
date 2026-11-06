"""Simple test opponents, not additional submission bots."""
import random
from bot7.cards import hand_class
from bot7.equity import PREFLOP, features, equity
from bot7.interface import Action, legalize, passive


class Baseline:
    def __init__(self, style="abc", seed=1):
        self.style = style
        self.rng = random.Random(seed)

    def act(self, obs):
        me = next(p for p in obs.players if p.seat == obs.hero)
        opponents = [p for p in obs.players if p.seat != obs.hero and not p.folded and p.stack + p.committed > 0]
        cost = obs.to_call
        pot = obs.eligible_pot if obs.eligible_pot is not None else obs.pot
        price = cost / max(1, pot + cost)
        raise_ok = "raise" in obs.legal_actions
        if self.style == "station":
            return Action("call" if cost else "check")
        if self.style == "random":
            r = self.rng.random()
            if raise_ok and r < .16:
                return legalize(obs, Action("raise", me.bet + cost + round(.7 * (pot + cost))))
            return Action("call" if cost and r < .72 else "fold" if cost else "check")
        if not obs.board:
            entry = PREFLOP[hand_class(*obs.hole)]
            pct = entry["percentile"]
            if raise_ok and pct < (.15 if self.style == "nit" else .27):
                if max(p.bet for p in obs.players) <= obs.big_blind or pct < .025:
                    return legalize(obs, Action("raise", max(3 * obs.big_blind, max(p.bet for p in obs.players) * 3)))
            if cost == 0:
                return Action("check")
            cutoff = .18 if self.style == "nit" else .36
            if pct < cutoff and cost <= .2 * me.stack:
                return Action("call")
            if pct < .03 and entry["equity"][4] > price + .08:
                return Action("call")
            return passive(obs)
        made, draw, wet = features(obs.hole, obs.board)
        if self.style == "equity":
            eq = equity(obs.hole, obs.board, [(1.0, 0)] * len(opponents), self.rng, 60)
            if raise_ok and eq > .67:
                return legalize(obs, Action("raise", me.bet + cost + round(.7 * (pot + cost))))
            return Action("call" if cost and eq > price + .05 else "fold" if cost else "check")
        if raise_ok and made > (.78 if self.style == "nit" else .62):
            return legalize(obs, Action("raise", me.bet + cost + round(.65 * (pot + cost))))
        estimate = max(.06, made - .12 * max(0, len(opponents) - 1)) + .4 * draw
        return Action("call" if cost and estimate > price + .13 else "fold" if cost else "check")

