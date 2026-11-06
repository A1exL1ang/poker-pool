"""Transparent local sparring opponents, not claims about contest opponents."""
from __future__ import annotations

import random
from equity import analyze, preflop_equities

REFERENCE_PARAMS = {
    "open_scale": 1.0, "call_margin": .025, "value_threshold": .70,
    "bluff_scale": 1.0, "range_selection": 1.0, "shove_margin": .025,
    "threebet_scale": 1.0, "realization": 1.0,
}


def _raise(obs, amount):
    if "raise" not in obs["legal_actions"]:
        return {"action": "call" if obs["to_call"] else "check"}
    maximum = obs["max_raise_to"]
    target = min(maximum, max(obs["min_raise_to"], int(amount)))
    return {"action": "raise", "amount": target}


class CallingStation:
    """Calls every price and checks every option; deliberately exploitable."""
    def reset(self):
        pass

    def act(self, obs):
        return {"action": "call" if obs["to_call"] else "check"}


class AlwaysFold:
    def reset(self):
        pass

    def act(self, obs):
        return {"action": "fold" if obs["to_call"] else "check"}


class RandomLegal:
    """Fuzzing adversary with a mix of checks, calls, folds and sizing extremes."""
    def __init__(self, seed=0):
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)

    def act(self, obs):
        action = self.rng.choice(obs["legal_actions"])
        if action != "raise":
            return {"action": action}
        maximum, minimum = obs["max_raise_to"], obs["min_raise_to"]
        target = maximum if maximum < minimum or self.rng.random() < .35 else self.rng.randint(minimum, maximum)
        return {"action": "raise", "amount": target}


class ShoveBot:
    """Push/fold opponent; threshold refers to HU random-range preflop equity."""
    def __init__(self, threshold=.57):
        self.threshold = threshold

    def reset(self):
        pass

    def act(self, obs):
        q = preflop_equities(obs["hole"])[-1] if not obs["board"] else analyze(obs["hole"], obs["board"])["equity_random"]
        if q >= self.threshold:
            if "raise" in obs["legal_actions"]:
                return {"action": "raise", "amount": obs["max_raise_to"]}
            return {"action": "call" if obs["to_call"] else "check"}
        return {"action": "fold" if obs["to_call"] else "check"}


class EquityBot:
    """Simple price-aware ABC baseline with fixed range/realization assumptions.

    Styles 'nit', 'tag', 'lag', and 'maniac' differ in opening breadth,
    bluff propensity and value thresholds. These are independent simple policies
    using the common hand-strength utility, not copied candidate logic.
    """
    _STYLE = {
        "nit": (.58, .72, .04, .14),
        "tag": (.52, .66, .10, .09),
        "lag": (.46, .62, .20, .04),
        "maniac": (.37, .56, .40, -.03),
    }

    def __init__(self, style="tag", seed=0, perturb=None):
        if style not in self._STYLE:
            raise ValueError("unknown local opponent style")
        self.style, self.seed = style, seed
        self.perturb = dict(perturb or {})
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)

    def act(self, obs):
        opening, value, bluff, safety = self._STYLE[self.style]
        opening += self.perturb.get("open_shift",0.)
        value += self.perturb.get("value_shift",0.)
        bluff *= self.perturb.get("bluff_scale",1.)
        safety += self.perturb.get("safety_shift",0.)
        call, pot, bb = obs["to_call"], max(1, obs["pot"]), obs["big_blind"]
        hero = obs["hero"]
        mine = obs["street_contributions"][hero]
        opponents = max(1, sum(obs["in_hand"]) - 1)
        if not obs["board"]:
            es = preflop_equities(obs["hole"])
            q = es[-1]
            raises = [h for h in obs["history"] if h["action"] == "raise"]
            if not raises:
                if q >= opening or (obs["button"] == hero and q >= opening - .06):
                    if "raise" in obs["legal_actions"]:
                        return _raise(obs, 3 * bb + max(0, pot - 1.5 * bb))
                    return {"action": "call" if call else "check"}
                return {"action": "fold" if call else "check"}
            tight = es[2] if self.style != "maniac" else es[4]
            effective = min(obs["stacks"][hero], max(s for i, s in enumerate(obs["stacks"]) if i != hero))
            if tight >= value + .03 and "raise" in obs["legal_actions"]:
                target = obs["max_raise_to"] if effective < 18 * bb else obs["current_bet"] * 3
                return _raise(obs, target)
            threshold = call / (pot + call) + safety
            if tight > threshold and q >= opening - .02 and call <= max(4 * bb, obs["stacks"][hero] * .25):
                return {"action": "call" if call else "check"}
            if call and tight > max(.53 + safety * .5, threshold + .03) and call >= obs["stacks"][hero]:
                return {"action": "call"}
            return {"action": "fold" if call else "check"}
        a = analyze(obs["hole"], obs["board"])
        q = (a["equity_random"] * .55 + a["equity_tight"] * .45) ** (opponents ** .58)
        price = call / (pot + call)
        if call:
            if q > value + .13 and "raise" in obs["legal_actions"]:
                return _raise(obs, mine + call + max(2 * call, .8 * (pot + call)))
            if q > price + safety and (q > .34 or a["draw_strength"] > .2):
                return {"action": "call"}
            return {"action": "fold"}
        if "raise" in obs["legal_actions"] and (q > value or
                (opponents == 1 and self.rng.random() < bluff and a["draw_strength"] > .08)):
            return _raise(obs, mine + max(bb, pot * (.65 if q > value else .4)))
        return {"action": "check"}


def make_lineup(kind="mixed", seed=0):
    """Seven opponents, deterministic style/seed specification for reproduction."""
    if kind == "mixed":
        return [EquityBot("nit", seed + 1), EquityBot("tag", seed + 2),
                EquityBot("tag", seed + 3), EquityBot("lag", seed + 4),
                CallingStation(), ShoveBot(.58), EquityBot("maniac", seed + 7)]
    if kind in EquityBot._STYLE:
        return [EquityBot(kind, seed + i + 1) for i in range(7)]
    if kind == "station":
        return [CallingStation() for _ in range(7)]
    if kind == "shove":
        return [ShoveBot(.52 + .02 * (i % 4)) for i in range(7)]
    if kind == "random":
        return [RandomLegal(seed + i + 1) for i in range(7)]
    if kind == "selfplay":
        from agent import Agent
        return [Agent(seed=seed+i+1,params=REFERENCE_PARAMS) for i in range(7)]
    if kind == "varied":
        rng = random.Random(seed ^ 0xBA512)
        opponents = []
        for i in range(7):
            choice = rng.random()
            if choice < .10:
                opponents.append(CallingStation())
            elif choice < .20:
                opponents.append(ShoveBot(rng.uniform(.49,.65)))
            else:
                style = rng.choice(("nit","tag","tag","lag","maniac"))
                perturb = dict(open_shift=rng.uniform(-.045,.045),
                               value_shift=rng.uniform(-.05,.05),
                               bluff_scale=rng.uniform(.4,1.9),
                               safety_shift=rng.uniform(-.065,.065))
                opponents.append(EquityBot(style,seed+i+1,perturb=perturb))
        return opponents
    raise ValueError("unknown local lineup")
