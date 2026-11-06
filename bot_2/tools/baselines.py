"""Independent, deliberately simple local opponents; not competition bots."""
from pokerbot.cards import evaluate
from pokerbot.equity import PERCENTILE
from pokerbot.cards import class_index
from pokerbot.model import legalize


class CheckFold:
    def act(self, obs):
        return legalize(obs, "check")


class CallingStation:
    def act(self, obs):
        return legalize(obs, "call")


class Shove:
    def act(self, obs):
        return legalize(obs, "raise", obs.max_raise_to)


class TightAggressive:
    """Fixed ranges and made-hand bets; no equity sampling or opponent model."""
    def __init__(self, loose=False):
        self.loose = loose

    def act(self, obs):
        hero = next(p for p in obs.players if p.seat == obs.hero)
        if not obs.board:
            pct = PERCENTILE[class_index(*obs.hole)]
            raised = max(p.street_bet for p in obs.players) > obs.big_blind
            cutoff = (.36 if self.loose else .20) if not raised else (.16 if self.loose else .09)
            if pct < (.06 if self.loose else .04):
                target = max(obs.min_raise_to, 3 * max(obs.big_blind, max(p.street_bet for p in obs.players)))
                if hero.stack < 14 * obs.big_blind:
                    target = obs.max_raise_to
                return legalize(obs, "raise", target)
            if pct < cutoff:
                if not raised:
                    return legalize(obs, "raise", 3 * obs.big_blind)
                if obs.to_call < .20 * hero.stack or pct < .075:
                    return legalize(obs, "call")
            return legalize(obs, "check")
        rank = evaluate(obs.hole + obs.board)
        cat = rank >> 20
        hole_top = max(c // 4 + 2 for c in obs.hole)
        top_pair = cat == 1 and (rank >> 16) & 15 == max(c // 4 + 2 for c in obs.board) and hole_top >= max(c // 4 + 2 for c in obs.board)
        if cat >= 2 or top_pair:
            return legalize(obs, "raise", hero.street_bet + obs.to_call + max(obs.big_blind, int(.65 * obs.pot)))
        if cat == 1 and obs.to_call < (.4 if self.loose else .20) * obs.pot:
            return legalize(obs, "call")
        if self.loose and obs.to_call < .10 * obs.pot:
            return legalize(obs, "call")
        return legalize(obs, "check")
