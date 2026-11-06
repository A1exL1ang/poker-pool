"""A bounded offline policy for the documented local NLHE interface.

This is an independently implemented range/equity policy, not a claimed solved
strategy. Native fixed-work equity is optional; the Python path is bounded.
"""

import json
from pathlib import Path
from secrets import randbits

from .cards import card, evaluate as python_evaluate, features, hand_class
from .icm import call_threshold
from .interface import legalize, safe_action


DEFAULTS = {
    "open_scale": 1.0,
    "call_margin": 0.018,
    "value_threshold": 0.61,
    "bluff_frequency": 0.12,
    "bet_scale": 0.68,
    "risk_scale": 1.0,
    "range_bluff": 0.08,
    "range_threshold_scale": 1.0,
    "samples": 96,
    "fewway_samples": 64,
    "multiway_samples": 20,
    "native": True,
    "adapt": True,
    "assume_unacted_continue": False,
    "probe_frequency": 0.40,
    "wide_threebet": True,
    "pushfold": False,
}


def _approx_preflop(key):
    from .cards import RANKS
    hi, lo = RANKS.index(key[0]), RANKS.index(key[1])
    if hi == lo:
        return 0.505 + hi * 0.0285
    suited = key[-1] == "s"
    gap = hi - lo
    return (0.30 + hi * 0.018 + lo * 0.012 - max(0, gap - 1) * 0.004
            + 0.026 * suited + 0.012 * (hi == 12))


def _starting_tables():
    from .cards import RANKS
    strengths = {}
    for hi in range(12, -1, -1):
        for lo in range(hi, -1, -1):
            for suffix in ([""] if hi == lo else ["s", "o"]):
                key = RANKS[hi] + RANKS[lo] + suffix
                strengths[key] = _approx_preflop(key)
    data = {}
    asset = Path(__file__).resolve().parent.parent / "assets" / "preflop.json"
    if asset.exists():
        with asset.open(encoding="utf-8") as f:
            data = json.load(f)
        source = data.get("hands", data.get("equities", data))
        for key in strengths:
            value = source.get(key) if isinstance(source, dict) else None
            if isinstance(value, (int, float)):
                strengths[key] = float(value)
            elif isinstance(value, dict):
                strengths[key] = float(value.get("equity", value.get("hu", value.get("uniform", value.get("random", [strengths[key]])[0]))))
    cumulative = 0.0
    percentiles = {}
    for key in sorted(strengths, key=lambda k: (-strengths[k], k)):
        combos = 6 if len(key) == 2 else 4 if key[-1] == "s" else 12
        percentiles[key] = (cumulative + combos * 0.5) / 1326.0
        cumulative += combos
    return strengths, percentiles, data


_STRENGTH, _TOP, _PREFLOP_DATA = _starting_tables()
_PUSHFOLD = None
_pushfold_path = Path(__file__).resolve().parent.parent / "assets" / "pushfold.json"
if _pushfold_path.exists():
    with _pushfold_path.open(encoding="utf-8") as _stream:
        _PUSHFOLD = json.load(_stream)
_PUSHFOLD_INDEX = {key: i for i, key in enumerate(_PUSHFOLD["classkeys"])} if _PUSHFOLD else {}
_DEPLOYMENT_CONFIG = {}
_policy_path = Path(__file__).resolve().parent.parent / "assets" / "policy.json"
if _policy_path.exists():
    with _policy_path.open(encoding="utf-8") as _stream:
        _policy_data = json.load(_stream)
    _DEPLOYMENT_CONFIG = _policy_data.get("config", _policy_data)


class _Opponent:
    __slots__ = ("hands", "vpip", "pfr", "bets", "checks", "folds", "width",
                 "threshold", "slope", "bluff", "draw", "seen_vpip", "seen_pfr", "seen_action")

    def __init__(self):
        self.hands = self.vpip = self.pfr = 0
        self.bets = self.checks = self.folds = 0
        self.reset_hand()

    def reset_hand(self):
        self.width = 1.0
        self.threshold = 0.0
        self.slope = 0.10
        self.bluff = 0.10
        self.draw = 0.18
        self.seen_vpip = self.seen_pfr = False
        self.seen_action = False

    @property
    def looseness(self):
        return (self.vpip + 10.0) / (self.hands + 30.0)

    @property
    def aggression(self):
        return (self.bets + 4.0) / (self.bets + self.checks + 12.0)


class Agent:
    """Instantiate a fresh Agent per table; call act(canonical_observation).

    No opponent identity or state is persisted between tables. Call reset()
    explicitly if an adapter reuses the instance. Configuration is immutable
    per instance and loaded before play. Runtime never opens files or sockets.
    """

    def __init__(self, seed=None, config=None):
        self.config = dict(DEFAULTS)
        self.config.update(_DEPLOYMENT_CONFIG)
        if config:
            self.config.update(config)
        for name in ("samples", "fewway_samples", "multiway_samples"):
            self.config[name] = min(256, max(8, int(self.config[name])))
        self.seed = (randbits(64) if seed is None else int(seed)) & ((1 << 64) - 1) or 13
        self.initial_seed = self.seed
        self.native = None
        if self.config["native"]:
            try:
                from .native import NativeEvaluator
                kernel = NativeEvaluator()
                if kernel.available:
                    self.native = kernel
                    if hasattr(kernel, "set_preflop_strength"):
                        kernel.set_preflop_strength(_STRENGTH)
            except (ImportError, OSError, AttributeError, RuntimeError, ValueError):
                self.native = None
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.initial_seed = int(seed) & ((1 << 64) - 1) or 13
        self.seed = self.initial_seed
        self.opponents = {}
        self.hand_id = None
        self.history_index = 0
        self.preflop_raises = 0
        self.last_aggressor = None
        self.street_bets = {}
        self.preflop_folded = set()
        self.starting_seats = set()
        self.decisions = self.fallbacks = self.native_errors = self.native_calls = 0

    def _random(self):
        x = self.seed
        x ^= (x << 13) & ((1 << 64) - 1)
        x ^= x >> 7
        x ^= (x << 17) & ((1 << 64) - 1)
        self.seed = x & ((1 << 64) - 1)
        return (self.seed >> 11) * (1.0 / (1 << 53))

    def _update(self, o):
        if self.hand_id != o["hand_id"]:
            self.hand_id = o["hand_id"]
            self.history_index = 0
            self.preflop_raises = 0
            self.last_aggressor = None
            self.street_bets = {}
            self.preflop_folded = set()
            self.starting_seats = {p["seat"] for p in o["players"] if p["stack"] + p["committed"] > 0}
            for p in o["players"]:
                model = self.opponents.setdefault(p["seat"], _Opponent())
                model.reset_hand()
        history = o.get("history", ())
        if len(history) < self.history_index:
            # An adapter must supply a complete hand history; tolerate truncation.
            self.history_index = 0
        for event in history[self.history_index:]:
            seat, street, action = event["seat"], event["street"], event["action"]
            model = self.opponents.setdefault(seat, _Opponent())
            if street == "preflop":
                # Only observed opportunities enter the denominator. Actions
                # occurring after hero folds are censored by this local API.
                if not model.seen_action:
                    model.hands += 1
                    model.seen_action = True
                if action in ("raise", "call"):
                    if not model.seen_vpip:
                        model.vpip += 1
                        model.seen_vpip = True
                if action == "raise":
                    self.preflop_raises += 1
                    self.last_aggressor = seat
                    if not model.seen_pfr:
                        model.pfr += 1
                        model.seen_pfr = True
                    active = len(self.starting_seats - self.preflop_folded)
                    size = float(event.get("amount", 0)) / max(1, o["big_blind"])
                    depth = float(event.get("stack_before", size * o["big_blind"])) / max(1, o["big_blind"])
                    base = (0.65 if active == 2 else 0.40 if active <= 4 else 0.25)
                    if self.config["adapt"]:
                        base *= max(0.65, min(1.6, (model.pfr + 6.0) / (model.hands + 30.0) / 0.20))
                    if self.preflop_raises > 1:
                        base = min(model.width * 0.50, 0.13 if self.preflop_raises == 2 else 0.055)
                    elif size > 4:
                        base *= max(0.24, (4.0 / size) ** 0.65)
                    if event.get("is_all_in") and depth <= 15:
                        base = max(base, (0.32 if active <= 3 else 0.16) + 0.10 * (depth < 6))
                    model.width = max(0.025, min(model.width, base))
                elif action == "call":
                    target = 0.55 if self.preflop_raises == 0 else 0.34 if self.preflop_raises == 1 else 0.12
                    if self.config["adapt"]:
                        target *= max(0.7, min(1.8, model.looseness / 0.33))
                    model.width = min(model.width, target)
                elif action == "fold":
                    self.preflop_folded.add(seat)
            else:
                before = max(o["big_blind"], float(event.get("pot_before", o["pot"])))
                paid = max(0.0, float(event.get("amount_paid", event.get("amount", 0))))
                frac = paid / before
                if action == "raise":
                    model.bets += 1
                    count = self.street_bets.get(street, 0) + 1
                    self.street_bets[street] = count
                    self.last_aggressor = seat
                    base = 0.42 + 0.24 * min(1.8, frac)
                    if count >= 2:
                        base += 0.12
                    if street == "river":
                        base += 0.06
                    if self.config["adapt"]:
                        base -= 0.22 * (model.aggression - 0.33)
                    model.threshold = max(model.threshold * 0.72, min(0.92, base * self.config["range_threshold_scale"]))
                    model.bluff = min(0.24, self.config["range_bluff"] + 0.13 * model.aggression)
                    model.draw = 0.50 if street == "flop" else 0.30 if street == "turn" else 0.0
                elif action == "call":
                    model.checks += 1
                    model.threshold = max(model.threshold * 0.70, min(0.72, 0.33 + frac * 0.23))
                    model.draw = 0.52 if street == "flop" else 0.32 if street == "turn" else 0.0
                elif action == "check":
                    model.checks += 1
                elif action == "fold":
                    model.folds += 1
        self.history_index = len(history)

    def act(self, observation):
        self.decisions += 1
        try:
            action = self._act(observation)
            return legalize(action, observation)
        except Exception:
            self.fallbacks += 1
            return safe_action(observation)

    decide = act

    def _act(self, o):
        self._update(o)
        hero = next(p for p in o["players"] if p["seat"] == o["hero"])
        hole = tuple(card(c) for c in o["hole_cards"])
        board = tuple(card(c) for c in o["board"])
        if len(hole) != 2 or len(set(hole + board)) != len(hole + board):
            raise ValueError("invalid visible cards")
        opponents = [p for p in o["players"] if p["seat"] != o["hero"] and not p["folded"] and p["stack"] + p["committed"] > 0]
        if not opponents:
            return {"action": "check"}
        cost = min(hero["stack"], max(0, int(o["to_call"])))
        if not board:
            return self._preflop(o, hero, opponents, hole, cost)
        return self._postflop(o, hero, opponents, hole, board, cost)

    def _position(self, o, hero, opponents):
        seats = sorted(p["seat"] for p in o["players"] if p["stack"] + p["committed"] > 0)
        size = max(o["button"], *(p["seat"] for p in o["players"])) + 1
        order = sorted(seats, key=lambda s: (s - o["button"] - 1) % size)
        # Last living player in clockwise order after the button acts last.
        live = {hero["seat"]} | {p["seat"] for p in opponents}
        return next((s for s in reversed(order) if s in live), hero["seat"]) == hero["seat"]

    def _raise(self, o, hero, target):
        if not o.get("can_raise", False):
            return {"action": "call" if o["to_call"] else "check"}
        return {"action": "raise", "amount": int(round(target))}

    def _preflop_vs_range(self, key, width):
        """Use trained conditional equity if present; conservative interpolation fallback."""
        data = _PREFLOP_DATA.get("hands", {}).get(key, {})
        conditional = data.get("vs_top", {}) if isinstance(data, dict) else {}
        if not conditional and isinstance(data, dict) and "ranges" in data:
            conditional = dict(zip(_PREFLOP_DATA["range_fractions"], data["ranges"]))
        if conditional:
            points = sorted((float(k), float(v)) for k, v in conditional.items())
            if width <= points[0][0]:
                return points[0][1]
            for (a, ea), (b, eb) in zip(points, points[1:]):
                if width <= b:
                    return ea + (eb - ea) * (width - a) / (b - a)
            return points[-1][1]
        raw = _STRENGTH[key]
        return max(0.13, raw - (1.0 - width) * (0.21 + 0.07 * (raw < 0.60)))

    def _preflop(self, o, hero, opponents, hole, cost):
        bb = max(1, int(o["big_blind"]))
        key = hand_class(*hole)
        top, hu = _TOP[key], _STRENGTH[key]
        depth = hero["stack"] / bb
        effective = min(hero["stack"] + hero["bet"], max(p["stack"] + p["bet"] for p in opponents)) / bb
        ip = self._position(o, hero, opponents)
        pot = o["pot"]
        raised = o["current_bet"] > bb
        # RFI widths by number of opponents still to act. Limps are isolated tighter.
        all_seats = sorted(p["seat"] for p in o["players"] if p["stack"] + p["committed"] > 0)
        size = max(o["button"], *(p["seat"] for p in o["players"])) + 1
        if len(all_seats) == 2:
            anchor = o["button"] - 1
        elif "big_blind_seat" in o:
            anchor = o["big_blind_seat"]
        else:
            clockwise = sorted(all_seats, key=lambda s: (s - o["button"] - 1) % size)
            anchor = clockwise[1]
        order = sorted(all_seats, key=lambda s: (s - anchor - 1) % size)
        behind = sum(p["seat"] in order[order.index(hero["seat"]) + 1:] and not p["folded"] and not p["all_in"] for p in opponents)
        limpers = sum(e["action"] == "call" for e in o.get("history", ()) if e["street"] == "preflop") if not raised else 0
        width = (0.82 if len(all_seats) == 2 else (0.75, 0.56, 0.46, 0.32, 0.25, 0.20, 0.17, 0.14)[min(7, behind)])
        width *= self.config["open_scale"]
        if self.config["pushfold"] and len(all_seats) == 2 and 2.0 <= effective <= 10.0 and _PUSHFOLD:
            endgame = self._short_hu(o, hero, opponents, key, effective)
            if endgame is not None:
                return endgame
        if limpers:
            width *= 0.73 / (1.0 + 0.10 * max(0, limpers - 1))
        if not raised and o.get("can_raise", False):
            if effective <= 12:
                # A one-step shove calculation: model folds and likely caller ranges.
                call_width = min(0.35, 0.07 + 0.75 / max(3.0, effective))
                if len(all_seats) == 2:
                    call_width = min(0.60, call_width + 0.13)
                active_behind = max(1, behind + limpers)
                folds = (1.0 - call_width) ** active_behind
                q = self._preflop_vs_range(key, call_width)
                multiple_penalty = max(0, active_behind - 1) * call_width * 0.035
                q = max(0, q - multiple_penalty)
                risk = min(hero["stack"], max(p["stack"] + max(0, p["bet"] - hero["bet"]) for p in opponents))
                future = min(risk, max(p["stack"] - max(0, hero["bet"] - p["bet"]) for p in opponents))
                shove_ev = folds * pot + (1 - folds) * (q * (pot + risk + future) - risk)
                if shove_ev > max(0.05 * bb, 0.01 * risk) and top < max(width, 0.15):
                    if self._risk_premium(o, hero, opponents, risk, pot) < 0.03 or q > 0.58 or depth < 5:
                        return self._raise(o, hero, hero["bet"] + hero["stack"])
                if cost == 0:
                    return {"action": "check"}
                if cost / max(1, hero["stack"]) <= 0.15 and top < width * 0.65 and effective > 8:
                    return self._raise(o, hero, max(o["min_raise_to"], 2.1 * bb))
                return {"action": "fold"}
            if top < width:
                target = (2.2 + min(3, limpers)) * bb
                if not ip and limpers:
                    target += 0.5 * bb
                if effective <= 22:
                    target = (2.05 + min(3, limpers)) * bb
                return self._raise(o, hero, max(o["min_raise_to"], target))
            if cost == 0:
                return {"action": "check"}
            # Complete small blind cheaply with playable hands; do not overlimp junk.
            if cost <= 0.55 * bb and top < min(0.66, width + 0.15) and effective > 18:
                return {"action": "call"}
            return {"action": "fold"}
        # Facing a raise: infer ranges from visible history and calculate pot odds.
        pending = 0
        if not self.config["assume_unacted_continue"]:
            involved = [p for p in opponents if p["bet"] > bb or p["all_in"] or p["seat"] == self.last_aggressor]
            if involved:
                pending = len(opponents) - len(involved)
                opponents = involved
        opponents.sort(key=lambda p: -(p["committed"] + p["stack"]))
        q, prefixes, _ = self._equity(o, hole, (), opponents)
        eligible = sum(min(p["committed"], hero["committed"] + cost) for p in o["players"])
        q = self._pot_equity(o, hero, opponents, cost, prefixes)
        odds = cost / max(1, eligible + cost)
        allin = cost == hero["stack"] or all(p["all_in"] for p in opponents)
        realization = 1.0 if allin else (0.96 if ip else 0.88)
        realization *= 1.0 - min(0.12, 0.022 * pending)
        if len(opponents) > 1 and not allin:
            realization -= min(0.10, 0.025 * (len(opponents) - 1))
        margin = self.config["call_margin"] + self._risk_premium(o, hero, opponents, cost, eligible)
        raise_count = max(1, self.preflop_raises)
        # Strong value reraises; a low-frequency suited blocker bluff heads-up.
        value_top = 0.035 if raise_count == 1 else 0.014 if raise_count == 2 else 0.006
        value_top *= min(1.65, max(0.65, 25.0 / max(15.0, effective)))
        if self.config["wide_threebet"] and raise_count == 1:
            opener = next((p for p in opponents if p["seat"] == self.last_aggressor), opponents[0])
            value_top *= min(2.2, max(0.7, self.opponents[opener["seat"]].width / 0.22))
        reraise = top < value_top and q > 0.48 + min(0.10, 0.025 * (len(opponents) - 1))
        bluff = (raise_count == 1 and len(opponents) == 1 and ip and effective >= 30
                 and key in ("A5s", "A4s", "A3s", "A2s", "KTs")
                 and self._random() < self.config["bluff_frequency"])
        if o.get("can_raise", False) and (reraise or bluff):
            if effective <= 20 or hero["stack"] <= 3.2 * cost or raise_count >= 3:
                return self._raise(o, hero, hero["bet"] + hero["stack"])
            target = (2.8 if ip else 3.5) * o["current_bet"]
            target += sum(max(0, p["bet"] - bb) for p in opponents if p["seat"] != self.last_aggressor) * 0.8
            return self._raise(o, hero, target)
        # Implied odds for small pairs and suited connectors only when deep/cheap.
        implied = (len(key) == 2 or (key[-1] == "s" and abs(hole[0] // 4 - hole[1] // 4) <= 2))
        implied_allowance = 0.035 if implied and effective > 35 and cost < 0.065 * hero["stack"] else 0.0
        return {"action": "call" if cost == 0 or q * realization + implied_allowance >= odds + margin else "fold"}

    def _short_hu(self, o, hero, opponents, key, depth):
        """Use only the precisely declared two-action HU abstraction context."""
        if len(opponents) != 1:
            return None
        history = [e for e in o.get("history", ()) if e["street"] == "preflop"]
        role = None
        if hero["seat"] == o["button"] and not history and o["current_bet"] <= o["big_blind"]:
            role = "shove_probs"
        elif (hero["seat"] != o["button"] and opponents[0]["all_in"] and len(history) == 1
              and history[0]["action"] == "raise"):
            model = self.opponents[opponents[0]["seat"]]
            if model.hands >= 15 and (model.pfr + 1) / (model.hands + 8) < 0.14:
                return None  # visible evidence of a much tighter-than-prior jammer
            role = "call_probs"
        if role is None:
            return None
        profiles = _PUSHFOLD["depths"]
        index = _PUSHFOLD_INDEX[key]
        lower, upper = profiles[0], profiles[-1]
        for profile in profiles:
            if profile["effective_stack_bb"] <= depth:
                lower = profile
            if profile["effective_stack_bb"] >= depth:
                upper = profile
                break
        a, b = lower["effective_stack_bb"], upper["effective_stack_bb"]
        probability = lower[role][index] if a == b else (lower[role][index] * (b - depth) + upper[role][index] * (depth - a)) / (b - a)
        if self._random() >= probability:
            return {"action": "fold"}
        return self._raise(o, hero, hero["bet"] + hero["stack"]) if role == "shove_probs" else {"action": "call"}

    def _equity(self, o, hole, board, opponents):
        models = [self.opponents[p["seat"]] for p in opponents]
        if self.native is not None:
            try:
                self.native_calls += 1
                specs = [(m.width, m.threshold, m.slope, m.bluff, m.draw) for m in models]
                budget = min(self.config["samples"], self.config["fewway_samples"] if len(models) in (2, 3)
                             else self.config["multiway_samples"] if len(models) >= 4 else 256)
                result = self.native.modeled_equity(hole, board, specs, seed=self.seed,
                                                     samples=budget)
                q = float(result["equity"])
                self.seed = (self.seed + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
                return q, tuple(result["prefixes"]), result
            except (ValueError, RuntimeError, OSError, AttributeError, TypeError):
                self.native_errors += 1
        # Bounded fallback: never run Python Monte Carlo under a 1ms contract.
        if not board:
            key = hand_class(*hole)
            qs = [self._preflop_vs_range(key, m.width) for m in models]
            prefixes, q = [], 1.0
            for v in qs:
                q *= v
                prefixes.append(q)
            return q, tuple(prefixes), {"strength": _STRENGTH[key], "stderr": 0.0}
        f = features(hole, board)
        score = python_evaluate(tuple(hole) + tuple(board))
        category = score >> 24
        br = [c // 4 + 2 for c in board]
        if category == 0:
            h = 0.26 + 0.07 * (f["high"] >= 13)
        elif category == 1:
            h = 0.78 if f["overpair"] else 0.72 if f["top_pair"] else 0.54 if f["pair_rank"] else 0.32
        elif category == 2:
            h = 0.84 if f["pair_rank"] else 0.43
        elif category == 3:
            h = 0.95 if f["pocket_set"] else 0.86 if f["pair_rank"] else 0.42
        else:
            h = {4: 0.93, 5: 0.97, 6: 0.985, 7: 0.997, 8: 0.999}[category]
            if len(board) == 5 and python_evaluate(board) == score:
                h = 0.5
        if len(board) < 5:
            h = min(0.99, h + 0.12 * f["flush_draw"] + 0.08 * f["open_ended"])
        prefixes, q = [], 1.0
        for m in models:
            conditional = max(0.01, min(0.99, h - m.threshold * max(0.05, (1 - h) * 0.35)))
            q *= conditional
            prefixes.append(q)
        return q, tuple(prefixes), {"strength": h, "stderr": 0.0}

    def _risk_premium(self, o, hero, opponents, cost, pot):
        payouts = o.get("payouts")
        if payouts is None or cost <= 0 or self.config["risk_scale"] <= 0:
            return 0.0
        live = [p for p in o["players"] if p["stack"] + p["committed"] > 0]
        if len(live) <= 2 or len(payouts) < len(live):
            return 0.0
        if all(float(payouts[i]) == float(payouts[1]) for i in range(2, len(live))):
            # Winner-take-all plus a constant is exactly linear in chip share.
            return 0.0
        # Dominant-opponent binary outcome: exact ICM of this simplified model.
        # Small routine pots use a local risk slope; expensive ICM is reserved for
        # material stack decisions and never invents missing payouts.
        if cost < 0.15 * max(1, hero["stack"]):
            spread = float(payouts[0]) - float(payouts[len(live) - 1])
            return self.config["risk_scale"] * (0.008 if spread > 0 else 0.0)
        if self.native is None and len(live) > 6:
            # Preserve the bounded Python fallback when no compiled ICM exists.
            # This conservative local proxy is not an exact ICM calculation.
            shorter = sum(p["stack"] + p["committed"] < 0.35 * (hero["stack"] + hero["committed"]) for p in live)
            return self.config["risk_scale"] * (0.02 + 0.04 * shorter / len(live)) * min(1.0, cost / max(1, hero["stack"]))
        hidx = next(i for i, p in enumerate(live) if p["seat"] == hero["seat"])
        villain = max(opponents, key=lambda p: p["committed"])
        vidx = next(i for i, p in enumerate(live) if p["seat"] == villain["seat"])
        base = [p["stack"] for p in live]
        total_pot = sum(p["committed"] for p in live)
        fold, win, lose = list(base), list(base), list(base)
        # Attribute contested chips to one dominant opponent. Other sidepots
        # remain with that opponent in both outcomes; total chips are conserved.
        fold[vidx] += total_pot
        risk = min(cost, base[hidx])
        win[hidx] += min(total_pot, pot)
        win[vidx] += max(0, total_pot - pot)
        lose[hidx] -= risk
        lose[vidx] += total_pot + risk
        odds = cost / max(1, pot + cost)
        if self.native is not None and hasattr(self.native, "icm_equity"):
            pf = self.native.icm_equity(fold, payouts[:len(live)], hidx)
            pw = self.native.icm_equity(win, payouts[:len(live)], hidx)
            pl = self.native.icm_equity(lose, payouts[:len(live)], hidx)
            threshold = max(0.0, min(1.0, (pf - pl) / (pw - pl))) if pw > pl else odds
        else:
            threshold = call_threshold(fold, win, lose, payouts[:len(live)], hidx, odds)
        return max(0.0, min(0.20, threshold - odds)) * self.config["risk_scale"]

    @staticmethod
    def _pot_equity(o, hero, opponents, cost, prefixes):
        """Expected share of hero-eligible chips, respecting nested side pots.

        Players who have chips to call remain possible contestants. Their future
        calls are not added as guaranteed profit. This is deliberately conservative.
        """
        cap = hero["committed"] + cost
        contributions = [min(cap, p["committed"] + (cost if p["seat"] == hero["seat"] else 0)) for p in o["players"]]
        levels = sorted(set(c for c in contributions if c > 0))
        last = award = total = 0.0
        capacities = [p["stack"] + p["committed"] for p in opponents]
        for level in levels:
            amount = (level - last) * sum(c >= level for c in contributions)
            n = sum(c >= level for c in capacities)
            share = prefixes[n - 1] if n else 1.0
            award += amount * share
            total += amount
            last = level
        return award / total if total > 0 else (prefixes[-1] if prefixes else 1.0)

    def _postflop(self, o, hero, opponents, hole, board, cost):
        bb = max(1, o["big_blind"])
        # Side pots have nested participant sets when sorted by total capacity.
        cap = hero["committed"] + cost
        opponents.sort(key=lambda p: -(p["stack"] + p["committed"]))
        q, prefixes, result = self._equity(o, hole, board, opponents)
        q = self._pot_equity(o, hero, opponents, cost, prefixes)
        f = features(hole, board)
        ip = self._position(o, hero, opponents)
        river = len(board) == 5
        eligible = sum(min(p["committed"], cap) for p in o["players"])
        pot = max(bb, eligible)
        odds = cost / max(1, pot + cost)
        allin = cost == hero["stack"] or all(p["all_in"] for p in opponents)
        realization = 1.0 if river or allin else (0.98 if ip else 0.92)
        if not river and not allin and len(opponents) > 1:
            realization -= 0.02 * min(3, len(opponents) - 1)
        if f["flush_draw"] or f["open_ended"]:
            realization = min(1.02, realization + 0.05)
        premium = self._risk_premium(o, hero, opponents, cost, pot)
        call_ev = q * realization - odds - premium - self.config["call_margin"]
        if cost and call_ev < 0:
            return {"action": "fold"}
        if not o.get("can_raise", False) or all(p["all_in"] for p in opponents):
            return {"action": "call" if cost else "check"}
        # Value bets need an edge against continuing ranges; a multiway pot
        # requires more strength and suppresses low-equity bluffs.
        threshold = self.config["value_threshold"] + min(0.10, 0.035 * (len(opponents) - 1))
        if cost:
            threshold += 0.10 + 0.03 * (cost > 0.75 * pot)
        if river:
            threshold -= 0.035 * ip
        value = q > threshold
        strength = float(result.get("strength", q))
        draw = f["flush_draw"] or f["open_ended"]
        was_aggressor = self.last_aggressor == hero["seat"]
        random = self._random()
        bluff_rate = self.config["bluff_frequency"]
        # A villain fold ending the hand produces no further act() observation.
        # Do not estimate fold frequency from that censored event stream.
        semi = (not river and draw and len(opponents) <= 2 and q > 0.31
                and random < min(0.62, bluff_rate * 3.0) and cost < 0.25 * pot)
        bluff = (not cost and len(opponents) == 1 and (ip or was_aggressor)
                 and q < 0.43 and (strength < 0.38 or f["nut_blocker"])
                 and random < bluff_rate * (1.5 if f["nut_blocker"] else 1.0))
        probe = (not cost and not river and len(opponents) <= 2 and (ip or was_aggressor)
                 and 0.26 < q < threshold
                 and random < self.config["probe_frequency"] * (0.55 if len(opponents) == 2 else 1.0)
                 * (0.75 if f["wet"] else 1.0))
        if value or semi or bluff or probe:
            frac = self.config["bet_scale"]
            if q < threshold + 0.10 and not f["wet"]:
                frac *= 0.63
            if f["wet"] and value:
                frac *= 1.15
            if q > 0.88:
                frac *= 1.18
            if probe and not value and not semi:
                frac = 0.33
            elif bluff:
                frac = 0.50 if river else 0.34
            if cost:
                target = o["current_bet"] + max(o["current_bet"] - hero["bet"], (pot + cost) * frac)
            else:
                target = hero["bet"] + max(bb, pot * frac)
            remaining = hero["stack"] - cost
            if q > max(0.70, threshold + 0.06) and remaining < (pot + cost) * 1.4:
                target = hero["bet"] + hero["stack"]
            return self._raise(o, hero, target)
        return {"action": "call" if cost else "check"}
