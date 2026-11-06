"""Original local no-limit Hold'em simulator and test opponents.

This is NOT the contest engine. The doubling blind schedule and optional
test payouts are explicitly synthetic. No information about other entries
is used. Invoke with python -m tools.simulate --help.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import time

from bot6.agent import Agent
from bot6.cards import evaluate, hand_class
from bot6.equity import PERCENTILE, preflop_equity
from bot6.interface import State, legalize
from bot6.icm import equity as icm_equity


def settle_pots(contributions, folded, scores, button=0):
    """Return gross awards, including refunds of unmatched contributions."""
    awards = [0] * len(contributions)
    levels = sorted(set(c for c in contributions if c > 0))
    previous = 0
    for level in levels:
        contributors = [i for i, c in enumerate(contributions) if c >= level]
        value = (level - previous) * len(contributors)
        previous = level
        eligible = [i for i in contributors if not folded[i]]
        if not eligible:
            # An unreachable state in valid betting; never silently destroy
            # chips if a test caller supplies it.
            raise ValueError("Side pot has no eligible player")
        best = max(scores[i] for i in eligible)
        winners = [i for i in eligible if scores[i] == best]
        each, extra = divmod(value, len(winners))
        for i in winners:
            awards[i] += each
        clockwise = sorted(winners, key=lambda i: (i - button - 1) % len(contributions))
        for i in clockwise[:extra]:
            awards[i] += 1
    return awards


class FixturePolicy:
    """Simple original validation opponents, not competitive submissions."""
    def __init__(self, style, seed):
        self.style, self.rng = style, random.Random(seed)

    def act(self, observation):
        s = State.from_dict(observation)
        style, roll = self.style, self.rng.random()
        if style == "fold":
            return legalize(s, "check" if not s.to_call else "fold")
        if style == "random":
            action = self.rng.choice(sorted(s.legal))
            return legalize(s, action, s.min_raise_to + self.rng.randrange(max(1, s.max_raise_to - s.min_raise_to + 1)))
        if not s.board:
            pct = PERCENTILE[hand_class(*s.hole)]
            if style == "maniac":
                if pct < .50 or roll < .25:
                    return legalize(s, "raise", max(s.min_raise_to, 4 * s.big_blind))
                return legalize(s, "call" if pct < .80 else "check" if not s.to_call else "fold")
            if style == "calling":
                return legalize(s, "call" if pct < .85 else "check" if not s.to_call else "fold")
            if pct < (.10 if style == "tight" else .17):
                return legalize(s, "raise", max(s.min_raise_to, 3 * s.big_blind))
            if pct < (.21 if style == "tight" else .38) and s.to_call <= 3 * s.big_blind:
                return legalize(s, "call")
            return legalize(s, "check" if not s.to_call else "fold")
        category = evaluate(s.hole + s.board) >> 24
        own_pair = any(a // 4 == b // 4 for a in s.hole for b in s.board) or s.hole[0] // 4 == s.hole[1] // 4
        strong = category >= 2 or category == 1 and own_pair
        if style == "maniac" and (strong or roll < .55):
            return legalize(s, "raise", s.me.bet + s.to_call + max(s.big_blind, s.pot * .8))
        if category >= 2 or strong and not s.to_call:
            return legalize(s, "raise", s.me.bet + s.to_call + max(s.big_blind, s.pot * .6))
        if style == "calling":
            return legalize(s, "call" if category >= 1 or s.to_call < s.pot * .35 else "check" if not s.to_call else "fold")
        if strong and s.to_call <= s.pot * (.5 if style == "tight" else .8):
            return legalize(s, "call")
        return legalize(s, "check" if not s.to_call else "fold")


class Table:
    def __init__(self, policies, seed=6, stacks=None, max_hands=200, payouts=(), table_id=None, capture=False, enforce_timeout=False):
        self.policies = policies
        self.n = len(policies)
        self.stacks = list(stacks or [2000] * self.n)
        self.initial_total = sum(self.stacks)
        self.rng = random.Random(seed)
        self.max_hands = max_hands
        self.payouts = tuple(payouts)
        self.button = seed % self.n
        self.table_id = table_id if table_id is not None else seed
        self.hand_number = 0
        self.eliminated = []
        self.events = Counter()
        self.decisions = []
        self.capture = capture
        self.enforce_timeout = enforce_timeout
        self.observations = []
        self.history = []

    def clockwise(self, start, seats):
        return sorted(seats, key=lambda x: (x - start - 1) % self.n)

    def _pay(self, seat, amount):
        amount = min(self.stacks[seat], amount)
        assert amount >= 0
        self.stacks[seat] -= amount
        self.bets[seat] += amount
        self.contributions[seat] += amount

    def _observation(self, seat, bb, highest, last_raise, can_raise):
        call = min(self.stacks[seat], max(0, highest - self.bets[seat]))
        legal = ["fold", "call"] if call else ["check"]
        maximum = self.bets[seat] + self.stacks[seat]
        other_can_call = any(not self.folded[i] and self.stacks[i] > max(0, highest - self.bets[i]) for i in self.live if i != seat)
        if can_raise and maximum > highest and self.stacks[seat] > call and other_can_call:
            legal.append("raise")
        obs = {
            "table_id": self.table_id, "hand_id": self.hand_number,
            "hero": seat, "hole_cards": self.holes[seat], "board": self.board[:],
            "button": self.button, "big_blind": bb, "pot": sum(self.contributions),
            "to_call": call, "min_raise_to": highest + last_raise,
            "max_raise_to": maximum, "legal_actions": legal,
            "players": [{"seat": i, "stack": self.stacks[i], "bet": self.bets[i],
                         "contributed": self.contributions[i], "folded": self.folded[i],
                         "all_in": self.stacks[i] == 0} for i in self.live],
            "history": [dict(h) for h in self.history],
        }
        if self.payouts:
            obs["payouts"] = self.payouts[:len(self.live)]
        return obs

    def _betting_round(self, start, bb, street):
        highest = max(self.bets)
        # The big blind remains the nominal preflop opening wager even if
        # posted short by an all-in player.
        if street == "preflop":
            highest = max(highest, bb)
        last_raise = bb
        pending = {i for i in self.live if not self.folded[i] and self.stacks[i] > 0}
        reopen_at = {i: 0 for i in pending}
        current = start
        actions = 0
        while pending:
            active = [i for i in self.live if not self.folded[i]]
            if len(active) <= 1:
                break
            able = [i for i in active if self.stacks[i] > 0]
            if len(able) == 1:
                # A short all-in blind does not force the only player with
                # chips to put money into an uncontested side pot.
                actual_other = max(self.bets[i] for i in active if i != able[0])
                highest = min(highest, max(self.bets[able[0]], actual_other))
                if self.bets[able[0]] >= highest:
                    break
            order = sorted(pending, key=lambda i: (i - current) % self.n)
            seat = order[0]
            pending.remove(seat)
            if self.folded[seat] or self.stacks[seat] <= 0:
                current = (seat + 1) % self.n
                continue
            can_raise = highest >= reopen_at[seat]
            obs = self._observation(seat, bb, highest, last_raise, can_raise)
            if self.capture and len(self.observations) < 5000:
                self.observations.append(obs)
            t0 = time.perf_counter_ns()
            result = self.policies[seat].act(obs)
            elapsed = time.perf_counter_ns() - t0
            if isinstance(self.policies[seat], Agent):
                self.decisions.append(elapsed)
                if self.enforce_timeout and elapsed > 1_000_000:
                    result = {"action": "check" if "check" in obs["legal_actions"] else "fold"}
                    self.events["timeout_fallback"] += 1
            action = result.get("action")
            if action not in obs["legal_actions"]:
                raise AssertionError(("Illegal action", action, obs, result))
            self.events[action] += 1
            if elapsed > 1_000_000 and isinstance(self.policies[seat], Agent):
                self.events["agent_over_1ms"] += 1
            before = sum(self.contributions)
            paid = 0
            if action == "fold":
                self.folded[seat] = True
            elif action == "call":
                paid = obs["to_call"]
                self._pay(seat, paid)
            elif action == "raise":
                target = result.get("amount")
                if not isinstance(target, int) or target <= highest or target > obs["max_raise_to"] or target < obs["min_raise_to"] and target != obs["max_raise_to"]:
                    raise AssertionError(("Illegal raise size", obs, result))
                old = highest
                paid = target - self.bets[seat]
                self._pay(seat, paid)
                highest = target
                if target - old >= last_raise:
                    last_raise = target - old
                    self.events["full_raise"] += 1
                else:
                    self.events["short_allin_raise"] += 1
                pending.update(i for i in active if i != seat and self.stacks[i] > 0 and self.bets[i] < highest)
            reopen_at[seat] = highest + last_raise
            self.history.append({"seat": seat, "street": street, "action": action,
                                 "amount": self.bets[seat], "paid": paid, "pot_before": before})
            current = (seat + 1) % self.n
            actions += 1
            if actions > 300:
                raise AssertionError("Betting round did not terminate")
        self._refund_uncalled()

    def _refund_uncalled(self):
        ordered = sorted(range(self.n), key=lambda i: self.bets[i], reverse=True)
        first, second = ordered[:2]
        excess = self.bets[first] - self.bets[second]
        if excess > 0:
            self.bets[first] -= excess
            self.contributions[first] -= excess
            self.stacks[first] += excess
            self.events["uncalled_refund"] += 1

    def play_hand(self, bb):
        self.live = [i for i, stack in enumerate(self.stacks) if stack > 0]
        if len(self.live) < 2:
            return
        if self.button not in self.live:
            self.button = self.clockwise(self.button, self.live)[0]
        before_stacks = self.stacks[:]
        self.bets, self.contributions = [0] * self.n, [0] * self.n
        self.folded = [i not in self.live for i in range(self.n)]
        deck = list(range(52))
        self.rng.shuffle(deck)
        self.holes = {i: [deck.pop(), deck.pop()] for i in self.live}
        self.board, self.history = [], []
        order = self.clockwise(self.button, self.live)
        if len(self.live) == 2:
            sb, big = self.button, order[0]
            first = sb
        else:
            sb, big = order[:2]
            first = order[2]
        self._pay(sb, max(1, bb // 2))
        self._pay(big, bb)
        self._betting_round(first, bb, "preflop")
        for street, count in (("flop", 3), ("turn", 1), ("river", 1)):
            if sum(not self.folded[i] for i in self.live) <= 1:
                break
            self.board.extend(deck.pop() for _ in range(count))
            self.bets = [0] * self.n
            self._betting_round(order[0], bb, street)
        remaining = [i for i in self.live if not self.folded[i]]
        if len(remaining) == 1:
            self.stacks[remaining[0]] += sum(self.contributions)
        else:
            assert len(self.board) == 5
            scores = [evaluate(tuple(self.holes[i]) + tuple(self.board)) if i in remaining else -1 for i in range(self.n)]
            awards = settle_pots(self.contributions, self.folded, scores, self.button)
            for i, amount in enumerate(awards):
                self.stacks[i] += amount
            self.events["showdown"] += 1
            self.events["sidepot_hand"] += len(set(self.contributions[i] for i in remaining)) > 1
        assert sum(self.stacks) == self.initial_total and min(self.stacks) >= 0
        busted = [i for i in self.live if self.stacks[i] == 0]
        # Ties in starting stack are randomly ordered solely for synthetic
        # rank reporting; real contest elimination tie rules are unknown.
        self.rng.shuffle(busted)
        busted.sort(key=lambda i: before_stacks[i])
        self.eliminated.extend(busted)
        self.hand_number += 1
        survivors = [i for i in self.live if self.stacks[i] > 0]
        if survivors:
            self.button = self.clockwise(self.button, survivors)[0]

    def run(self):
        while sum(s > 0 for s in self.stacks) > 1 and self.hand_number < self.max_hands:
            # Deliberately synthetic: rules give 10/20 initially and an
            # increase every 25 hands, but no sequence of later blind sizes.
            bb = 20 * (2 ** (self.hand_number // 25))
            self.play_hand(bb)
        return self.stacks


def percentile(values, pct):
    data = sorted(values)
    return data[min(len(data)-1, int(pct * len(data)))] if data else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6006)
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--field", choices=["mixed", "tight", "calling", "maniac", "self"], default="mixed")
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--time-budget-us", type=int, default=500)
    parser.add_argument("--output", default="benchmarks/simulation.json")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--enforce-timeout", action="store_true", help="Apply the rules' fold/check fallback above 1 ms on this host")
    args = parser.parse_args()
    styles = ["tight", "calling", "standard", "maniac", "tight", "calling", "standard"]
    records, all_latency, captured = [], [], []
    events = Counter()
    start = time.perf_counter()
    for t in range(args.tables):
        hero = t % 8
        opponents = iter(styles)
        policies = []
        for i in range(8):
            if i == hero or args.field == "self":
                policies.append(Agent(seed=args.seed + t * 17 + i, sample_budget=args.budget, time_budget_us=args.time_budget_us))
            else:
                style = next(opponents) if args.field == "mixed" else args.field
                policies.append(FixturePolicy(style, args.seed + t * 19 + i))
        table = Table(policies, args.seed + t, max_hands=args.hands, capture=args.capture and t < 12, enforce_timeout=args.enforce_timeout)
        table.run()
        chipshare = table.stacks[hero] / 16000
        # WTA ICM at a cap equals chip share; no contest payouts assumed.
        rank = 8 - table.eliminated.index(hero) if hero in table.eliminated else None
        records.append({"table": t, "hero": hero, "chips": table.stacks[hero], "chip_share": chipshare,
                        "finished_rank": rank, "outright_win": table.stacks[hero] == 16000,
                        "hands": table.hand_number, "survivors": sum(s > 0 for s in table.stacks)})
        events.update(table.events)
        all_latency.extend(table.decisions)
        captured.extend(table.observations)
        if (t + 1) % 25 == 0:
            print(f"{t+1}/{args.tables}: mean final chip share={statistics.mean(r['chip_share'] for r in records):.4f}", flush=True)
    shares = [r["chip_share"] for r in records]
    summary = {
        "notice": "Synthetic local tests against original simple fixtures; not official contest performance.",
        "seed": args.seed, "field": args.field, "tables": args.tables,
        "schedule": "synthetic: 20 chip BB, doubled every 25 hands", "payout_assumption": "winner-take-all diagnostic only",
        "sample_budget": args.budget, "elapsed_seconds": time.perf_counter() - start,
        "time_budget_us": args.time_budget_us,
        "enforce_1ms_timeout": args.enforce_timeout,
        "mean_final_chip_share": statistics.mean(shares),
        "standard_error_chip_share": statistics.stdev(shares) / len(shares) ** .5 if len(shares) > 1 else 0,
        "outright_win_rate": statistics.mean(r["outright_win"] for r in records),
        "agent_decisions": len(all_latency), "latency_us": {"median": percentile(all_latency, .5) / 1000,
          "p95": percentile(all_latency, .95) / 1000, "p99": percentile(all_latency, .99) / 1000,
          "max": max(all_latency, default=0) / 1000}, "events": dict(events), "records": records,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    if args.capture:
        Path("benchmarks/states.jsonl").write_text("".join(json.dumps(o, separators=(",", ":")) + "\n" for o in captured))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
