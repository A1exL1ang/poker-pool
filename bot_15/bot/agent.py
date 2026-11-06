"""Bounded, offline, range-aware tournament policy for the LOCAL interface.

This is an independently designed approximate policy, not a solved equilibrium.
No decision-time sampling, network calls, or external runtime dependencies.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random

from .icm import hero_equity
from .protocol import legalize, safe_action


@dataclass(frozen=True)
class Config:
    open_scale: float = 1.0
    call_margin: float = 0.025
    aggression: float = 1.0
    range_pressure: float = 1.0
    bluff_scale: float = 1.0
    icm_scale: float = 1.0
    adaptation: bool = True


_SELECTED_CONFIG = None


def selected_config():
    """Load the frozen local selection once during construction, not act()."""
    global _SELECTED_CONFIG
    if _SELECTED_CONFIG is None:
        path = Path(__file__).resolve().parent.parent / 'artifacts' / 'policy.json'
        if not path.exists():
            raise RuntimeError('Policy asset missing: package artifacts/policy.json')
        values = json.loads(path.read_text())['config']
        cfg = Config(**values)
        if not (.5 <= cfg.open_scale <= 2 and -.1 <= cfg.call_margin <= .2
                and 0 <= cfg.range_pressure <= 2 and 0 <= cfg.bluff_scale <= 3
                and 0 <= cfg.icm_scale <= 2 and .1 <= cfg.aggression <= 2
                and type(cfg.adaptation) is bool):
            raise ValueError('Policy asset has invalid parameters')
        _SELECTED_CONFIG = cfg
    return _SELECTED_CONFIG


class Agent:
    def __init__(self, seed=15, config=None):
        # Import/load assets before the timed gameplay path.
        from .equity import estimate, preflop_percentile, forced_chop, require_assets
        require_assets()
        self.estimate = estimate
        self.percentile = preflop_percentile
        self.forced_chop = forced_chop
        self.config = config if config is not None else selected_config()
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = Random(self.seed)
        self.hand_id = None
        self.seen = 0
        self.first = set()
        # [preflop first decisions, voluntarily entered, raises, postflop
        # facing-bet decisions, folds, postflop actions, aggressive actions]
        self.stats = [[0.0] * 7 for _ in range(8)]
        self.errors = 0
        self.decisions = 0

    def _observe(self, o):
        hand_id = o.get("hand_id", 0)
        if hand_id != self.hand_id:
            self.hand_id = hand_id
            self.seen = 0
            self.first.clear()
        history = o.get("history", ())
        if len(history) < self.seen:
            self.seen = 0
        # Legitimate NLHE histories are short; bound adversarial adapter input.
        for row in history[self.seen:self.seen + 128]:
            seat = row["seat"]
            if seat == o["hero"] or not 0 <= seat < 8:
                continue
            stats = self.stats[seat]
            kind = row["action"]
            if row["street"] == 0:
                if seat not in self.first:
                    self.first.add(seat)
                    stats[0] += 1
                    stats[1] += kind in ("call", "raise")
                    stats[2] += kind == "raise"
            else:
                stats[5] += 1
                stats[6] += kind == "raise"
                if row.get("to_call", 0) > 0:
                    stats[3] += 1
                    stats[4] += kind == "fold"
        self.seen = len(history)

    def _profile(self, seat):
        if not self.config.adaptation:
            return 0.28, 0.17, 0.45, 0.32
        s = self.stats[seat]
        # Conservative pseudo-counts avoid hard reads on a few hands. Seat-local
        # statistics are discarded at reset; no opponent/team identities.
        return ((s[1] + 8.4) / (s[0] + 30),
                (s[2] + 5.1) / (s[0] + 30),
                (s[4] + 6.75) / (s[3] + 15),
                (s[6] + 6.4) / (s[5] + 20))

    @staticmethod
    def _position(o, opponents):
        n, hero, button = len(o["stacks"]), o["hero"], o["button"]
        # Postflop position determined clockwise from button; button acts last.
        dist = lambda seat: (seat - button - 1) % n
        ip = all(dist(hero) > dist(s) for s in opponents)
        if "position" in o:
            behind = o["position"]
        else:
            alive = [s for s in range(n) if o["stacks"][s] + o["contributions"][s] > 0]
            clockwise = sorted(alive, key=lambda s: (s - button) % n)
            if len(alive) == 2:
                order = clockwise
            else:
                offset = 3 if button in alive else 2
                order = clockwise[offset:] + clockwise[:offset]
            behind = sum(s in opponents for s in order[order.index(hero) + 1:])
        return behind, ip

    @staticmethod
    def _pot_equity(o, cost, raw, versus_bettor, aggressor):
        """Expected share weighted by eligible pot layers (approximate ranges).

        This prevents a tiny all-in hand depressing equity in a large side pot
        it cannot win. Future calls/raises remain covered only by realization.
        """
        h = o["hero"]
        contributions = o["contributions"][:]
        contributions[h] += cost
        cap = contributions[h]
        total = expected = previous = 0.0
        for level in sorted({min(c, cap) for c in contributions} - {0}):
            contributors = [s for s, c in enumerate(contributions) if c >= level]
            rivals = [s for s in contributors if s != h and o["in_hand"][s]]
            size = (level - previous) * len(contributors)
            if aggressor in rivals:
                eq = versus_bettor * raw ** (max(0, len(rivals) - 1) * 0.85)
            else:
                eq = raw ** (len(rivals) * 0.85 + 0.15) if rivals else 1.0
            expected += size * eq
            total += size
            previous = level
        return expected / total if total else raw

    @staticmethod
    def _texture(hole, board):
        ranks = [c // 4 for c in hole + board]
        suits = [c % 4 for c in hole + board]
        branks = [c // 4 for c in board]
        bsuits = [c % 4 for c in board]
        flushdraw = any(suits.count(s) == 4 and any(c % 4 == s for c in hole)
                        for s in range(4))
        mask = 0
        for r in ranks:
            mask |= 1 << (r + 1)
            if r == 12:
                mask |= 1
        straightdraw = any((mask & (31 << lo)).bit_count() == 4 for lo in range(10))
        wet = max((bsuits.count(s) for s in range(4)), default=0) >= 3
        connected = any(sum(lo <= r <= lo + 4 for r in set(branks)) >= 3 for lo in range(9))
        paired = len(set(branks)) != len(branks)
        draw = len(board) < 5 and (flushdraw or straightdraw)
        return draw, wet or connected, paired

    def _risk_threshold(self, o, cost, eligible, opponent, chip_threshold):
        """ICM adjustment for a binary decisive showdown approximation.

        All-in HU outcomes are chip-accounted including uncallable excess.
        Multiway outcomes have an unresolved side-pot model: use only a capped
        risk adjustment, never pretend this is exact multiway ICM search.
        """
        payouts = o.get("payouts")
        if not payouts or not self.config.icm_scale or cost <= 0:
            return chip_threshold
        h = o["hero"]
        alive = [s for s in range(len(o["stacks"]))
                 if o["stacks"][s] + o["contributions"][s] > 0]
        if len(alive) <= 2 or cost < 0.2 * o["stacks"][h]:
            return chip_threshold
        # Only actual remaining prizes, no invented payout vector.
        prizes = payouts[:len(alive)]
        if len(prizes) < len(alive) or prizes[1:] == [prizes[-1]] * (len(prizes) - 1):
            return chip_threshold
        index = alive.index(h)
        villain = alive.index(opponent)
        base = [float(o["stacks"][s]) for s in alive]
        # Fold: chosen opponent receives pot. Call-win: receive eligible pot
        # plus own call; any uncallable pot remainder goes to that opponent.
        fold = base[:]
        fold[villain] += o["pot"]
        lose = fold[:]
        lose[index] -= cost
        lose[villain] += cost
        win = base[:]
        win[index] += eligible
        win[villain] += o["pot"] - eligible
        u0 = hero_equity(fold, prizes, index)
        uw = hero_equity(win, prizes, index)
        ul = hero_equity(lose, prizes, index)
        if uw <= ul:
            return chip_threshold
        required = (u0 - ul) / (uw - ul)
        premium = max(0.0, min(0.22, required - chip_threshold))
        return min(0.95, chip_threshold + premium * self.config.icm_scale)

    def act(self, observation):
        self.decisions += 1
        try:
            self._observe(observation)
            return legalize(observation, self._act(observation))
        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError):
            self.errors += 1
            return safe_action(observation)

    def _act(self, o):
        h, stacks = o["hero"], o["stacks"]
        stack = stacks[h]
        opponents = [s for s, live in enumerate(o["in_hand"]) if live and s != h]
        if stack <= 0 or not opponents:
            return safe_action(o)
        hole, board = list(o["hole"]), list(o["board"])
        bb = max(1, o["big_blind"])
        cost = min(stack, max(0, o["to_call"]))
        current, maximum = o["street_bets"][h], o["max_raise_to"]
        # Exclude pots hero cannot win; folded chips remain eligible by level.
        cap = o["contributions"][h] + cost
        eligible = sum(min(c, cap) for c in o["contributions"])
        pot = o["pot"]
        behind, ip = self._position(o, opponents)
        history = o.get("history", ())[-128:]
        raises = [r for r in history if r["action"] == "raise"]
        pre_raises = [r for r in raises if r["street"] == 0]
        active_raises = [r for r in raises if r["street"] == o["street"]]
        aggressor = next((r["seat"] for r in reversed(active_raises) if r["seat"] != h),
                         max(opponents, key=lambda s: o["street_bets"][s]))
        vpip, pfr, fold_rate, aggr = self._profile(aggressor)
        threshold = cost / max(1, eligible + cost)
        threshold = self._risk_threshold(o, cost, eligible, aggressor, threshold)
        if not board:
            return self._preflop(o, hole, opponents, behind, ip, pre_raises,
                                 vpip, pfr, threshold, cost, eligible)
        # Infer preflop range of strongest still-live opponent. Weak priors and
        # soft tightening intentionally avoid treating a guess as known cards.
        active_pr = [r for r in pre_raises if r["seat"] in opponents]
        if active_pr:
            _, opener_pfr, _, _ = self._profile(active_pr[-1]["seat"])
            fraction = max(0.045, min(0.65, opener_pfr))
            if len(pre_raises) >= 2:
                fraction *= 0.48 ** min(2, len(pre_raises) - 1)
        else:
            fraction = max(0.25, min(1.0, vpip * 1.8))
            if len(opponents) == 1 and not pre_raises:
                fraction = max(fraction, 0.65)
        raw = self.estimate(hole, board, max(0.03, fraction))
        draw, wet, paired = self._texture(hole, board)
        q = 1.0
        if cost:
            size = cost / max(bb, eligible - cost)
            q = (0.72, 0.56, 0.43)[len(board) - 3]
            q += (aggr - 0.32) * 0.85
            q -= 0.12 * max(0, min(1.5, size - 0.4))
            q *= 0.72 ** max(0, len(active_raises) - 1)
            prior_bets = sum(r["seat"] == aggressor and 0 < r["street"] < o["street"] for r in raises)
            q *= 0.88 ** prior_bets
            q = max(0.16, min(0.95, 1 - (1 - q) * self.config.range_pressure))
        # River: top-q truncation of strength percentile; earlier streets retain
        # some draw/runout equity, since equity is not current hand rank.
        truncated = max(0.0, (raw - (1 - q)) / q)
        blend = 1.0 if len(board) == 5 else (0.53 if draw else 0.70)
        versus_bettor = raw * (1 - blend) + truncated * blend
        equity = self._pot_equity(o, cost, raw, versus_bettor, aggressor)
        # A royal-flush board is an exact tie for every remaining player. Avoid
        # treating tied-board equity as the bottom of an aggressive range.
        if self.forced_chop(board):
            return {"action": "call" if cost else "check", "amount": cost}
        realization = 1.0 if len(board) == 5 or cost >= stack or all(stacks[s] == 0 for s in opponents) else (0.95 if ip else 0.88)
        if draw and cost < stack:
            realization = min(1.0, realization + 0.035)
        margin = 0 if cost >= stack or len(board) == 5 else self.config.call_margin
        can_call = equity * realization >= threshold + margin
        if not o.get("can_raise") or all(stacks[s] == 0 for s in opponents):
            return {"action": "call" if cost and can_call else "check" if not cost else "fold", "amount": cost}
        effective = min(stack, max(stacks[s] for s in opponents))
        spr = effective / max(bb, pot)
        # Value raising must beat a stronger calling range than the betting range.
        value_threshold = (0.76 if cost else 0.66) + 0.035 * min(3, len(opponents) - 1)
        if spr < 1.5:
            value_threshold -= 0.07
        value_threshold += 0.03 * (1 - self.config.aggression)
        value = equity >= value_threshold
        # Semi-bluffs and occasional low-showdown stabs; no multiway pure bluffs.
        ours_pre = bool(pre_raises and pre_raises[-1]["seat"] == h)
        checks = sum(r["street"] == o["street"] and r["action"] == "check" for r in history)
        stab = (0.15 + 0.22 * (fold_rate - 0.45) + 0.10 * ip + 0.07 * ours_pre)
        if wet:
            stab *= 0.65
        if cost:
            stab = 0.11 if draw and raw > 0.38 and len(opponents) == 1 else 0.0
        elif len(opponents) > 1:
            stab = 0.07 if draw and len(opponents) == 2 else 0.0
        elif not draw and raw > 0.48:
            stab *= 0.25  # preserve medium-strength hands for showdown
        elif len(board) == 5:
            stab *= 0.65
        if not cost and not ip and not ours_pre and checks == 0:
            stab *= 0.25
        bluff = self.rng.random() < stab * self.config.bluff_scale and spr > 1.0
        if value or bluff:
            fraction = 0.70 if wet or equity > 0.90 else 0.48
            if not value:
                fraction = 0.44 if not cost else 0.66
            target = current + cost + round(fraction * (pot + cost))
            if value and (target - current >= effective * 0.62 or spr < 1.2):
                target = maximum
            if cost and not value and target - current > stack * 0.48:
                bluff = False
            if value or bluff:
                return {"action": "raise", "amount": target}
        return {"action": "call" if cost and can_call else "check" if not cost else "fold", "amount": cost}

    def _preflop(self, o, hole, opponents, behind, ip, raises, vpip, pfr, threshold, cost, eligible):
        h = o["hero"]
        stack, bb = o["stacks"][h], o["big_blind"]
        current = o["street_bets"][h]
        rank = self.percentile(hole)
        effective = min(stack + current, max(o["stacks"][s] + o["street_bets"][s] for s in opponents))
        depth = effective / bb
        can_raise = o.get("can_raise", False)
        if not can_raise and cost and all(o["stacks"][s] == 0 for s in opponents):
            # Blinds alone can put opponents all-in without any raise event.
            # Evaluate eligibility per pot layer; very short players cannot win
            # the deeper side pot. Independence approximation only for equity.
            cap = o["contributions"][h] + cost
            final_contributions = o["contributions"][:]
            final_contributions[h] += cost
            previous = 0
            expected = 0.0
            for level in sorted({min(c, cap) for c in final_contributions} - {0}):
                contributors = [s for s, c in enumerate(final_contributions) if c >= level]
                rivals = sum(s in opponents for s in contributors)
                layer = (level - previous) * len(contributors)
                expected += layer * self.estimate(hole, [], 1.0) ** rivals
                previous = level
            can_call = expected / max(1, eligible + cost) >= threshold
            return {"action": "call" if can_call else "fold", "amount": cost}
        # Unraised pots: opening width responds to seats still behind, plus
        # limpers. Short stacks use an explicitly approximated shove EV.
        if not raises:
            limpers = sum(r["street"] == 0 and r["action"] == "call" and r["seat"] != h for r in o.get("history", ())[-128:])
            width = max(0.14, min(0.72, 0.57 - 0.065 * max(0, behind - 1)))
            if len(opponents) == 1:
                width = 0.78
            width = max(0.08, min(0.95, width * self.config.open_scale / (1 + 0.13 * limpers)))
            if depth <= 11 and can_raise:
                # At most seven preflop table lookups; one-caller approximation
                # gets conservative multiway and uncertainty deductions.
                no_call = 1.0
                mean_eq = 0.0
                weight = 0.0
                for seat in opponents:
                    v, _, _, _ = self._profile(seat)
                    call_range = max(0.09, min(0.50, 0.09 + 0.21 * v + 0.65 / max(2, depth)))
                    no_call *= 1 - call_range
                    mean_eq += call_range * self.estimate(hole, [], call_range)
                    weight += call_range
                mean_eq /= max(0.001, weight)
                risk = max(0, min(stack, effective - current))
                ev = no_call * o["pot"] + (1 - no_call) * (mean_eq * (o["pot"] + 2 * risk) - risk)
                ev -= max(0, len(opponents) - 1) * 0.025 * risk
                # Payout pressure applies when supplied, via call threshold
                # premium used as a conservative extra all-in risk charge.
                ev -= max(0, threshold - cost / max(1, eligible + cost)) * risk
                if ev > 0.08 * bb and rank < max(width, 0.28):
                    return {"action": "raise", "amount": o["max_raise_to"]}
            if rank < width and can_raise:
                size = (2.25 if depth >= 30 else 2.05) + limpers * 0.9
                if not ip and limpers:
                    size += 0.5
                if depth <= 7:
                    # Marginal hands that failed shove EV are checked/folded.
                    return {"action": "check" if not cost else "fold", "amount": 0}
                return {"action": "raise", "amount": round(size * bb)}
            if not cost:
                return {"action": "check", "amount": 0}
            # Cheap small-blind completes and multiway speculative overlimps.
            eq = self.estimate(hole, [], max(0.30, vpip)) ** (len(opponents) * 0.75 + 0.25)
            suited = hole[0] % 4 == hole[1] % 4
            pair = hole[0] // 4 == hole[1] // 4
            if cost <= 0.55 * bb and rank < 0.70 and eq * 0.75 > threshold + 0.03:
                return {"action": "call", "amount": cost}
            if limpers >= 2 and depth > 18 and (suited or pair) and rank < 0.5 and eq > threshold + 0.07:
                return {"action": "call", "amount": cost}
            return {"action": "fold", "amount": 0}
        # Facing a raise: range is based on observed seat behavior, sizing and
        # raise count, with soft bounds to avoid impossible reads.
        fraction = max(0.06, min(0.65, pfr))
        last = raises[-1]
        if len(raises) == 1:
            size = last["amount"] / bb
            if size >= 6:
                fraction *= 0.68 if depth > 16 else 1.15
            if behind <= 1 and len(opponents) <= 2:
                fraction = min(0.70, fraction * 1.45)
        else:
            fraction *= 0.43 ** min(2, len(raises) - 1)
        fraction = max(0.025, fraction)
        eq = self.estimate(hole, [], fraction)
        # Already participating callers also contest the pot; not-yet-acting
        # players are addressed through realization instead of treating all as
        # guaranteed showdown opponents.
        contenders = sum(o["street_bets"][s] > bb or o["stacks"][s] == 0 for s in opponents)
        eq_multi = eq ** (1 + max(0, contenders - 1) * 0.78)
        allin = cost >= stack or all(o["stacks"][s] == 0 for s in opponents)
        realization = 1.0 if allin else (0.94 if ip else 0.81)
        if hole[0] % 4 == hole[1] % 4:
            realization += 0.025 if not allin else 0
        value_width = max(0.018, min(0.085, fraction * 0.22))
        value = rank < value_width and eq_multi > (0.55 if depth < 25 else 0.58)
        if can_raise and value:
            target = round(last["amount"] * (3.0 if ip else 3.6))
            if depth < 22 or target >= effective * 0.45:
                target = o["max_raise_to"]
            return {"action": "raise", "amount": target}
        # Rare suited-ace blocker three-bets against a wide opening range.
        ace_suited = max(c // 4 for c in hole) == 12 and hole[0] % 4 == hole[1] % 4
        if can_raise and len(raises) == 1 and depth > 35 and len(opponents) <= 2 and ace_suited and fraction > 0.22 and self.rng.random() < 0.10 * self.config.bluff_scale:
            return {"action": "raise", "amount": round(last["amount"] * (3 if ip else 3.6))}
        margin = 0 if allin else self.config.call_margin
        can_call = eq_multi * realization >= threshold + margin
        # Set-mining is a limited implied-odds exception, not an automatic call.
        pair = hole[0] // 4 == hole[1] // 4
        if pair and cost < effective / 18 and cost <= 4 * bb and len(raises) == 1:
            can_call = True
        return {"action": "call" if can_call else "check" if not cost else "fold", "amount": cost}
