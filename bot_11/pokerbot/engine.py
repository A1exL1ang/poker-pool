"""Independent, local no-limit Hold'em test engine.

This is not the contest engine or its interface. The blind growth is configurable;
doubling every 25 hands is a deliberately labelled local assumption. No payouts
are supplied by default. Card ids are rank * 4 + suit, with 2 at rank zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import random
from time import perf_counter_ns
from typing import Callable, Sequence

from .evaluator import evaluate


STREETS = ("preflop", "flop", "turn", "river")


@dataclass(slots=True)
class Player:
    seat: int
    stack: int
    street_bet: int = 0
    committed: int = 0
    folded: bool = False
    all_in: bool = False
    cards: tuple[int, int] = ()


def icm_values(stacks: Sequence[int], payouts: Sequence[float]) -> list[float]:
    """Exact independent-chip-model values; payouts ordered first to last.

    Zero-stack players receive zero here. Previously eliminated prizes must be
    handled separately by the tournament. Complexity O(n 2**n), n <= 8 here.
    """
    live = [(i, s) for i, s in enumerate(stacks) if s > 0]
    n = len(live)
    if len(payouts) < n:
        raise ValueError("need one payout for each surviving player")
    if any(not isfinite(float(x)) for x in payouts):
        raise ValueError("payouts must be finite")
    out = [0.0] * len(stacks)
    if not n:
        return out
    weights = [float(s) for _, s in live]
    full = (1 << n) - 1

    probabilities = {full: 1.0}
    for place in range(n):
        following: dict[int, float] = {}
        award = float(payouts[place])
        for mask, probability in probabilities.items():
            total = sum(weights[i] for i in range(n) if mask & (1 << i))
            for i in range(n):
                bit = 1 << i
                if mask & bit:
                    p = probability * weights[i] / total
                    out[live[i][0]] += p * award
                    rest = mask ^ bit
                    if rest:
                        following[rest] = following.get(rest, 0.0) + p
        probabilities = following
    return out


class Hand:
    """One hand. Deck injection and explicit stack lists support mechanics tests.

    Deck cards are consumed from index zero: two dealing rounds beginning left
    of the button, then burn/flop/burn/turn/burn/river. Every agent sees only its
    own cards. Blinds and all actions are recorded in history.
    """

    def __init__(self, stacks: Sequence[int], agents: Sequence[Callable], *,
                 button: int = 0, small_blind: int = 10, big_blind: int = 20,
                 hand_number: int = 0, rng: random.Random | None = None,
                 deck: Sequence[int] | None = None, payouts=None,
                 enforce_time_limit: bool = False, decision_limit_ns=1_000_000,
                 record_observations: bool = False, blind_seats=None):
        if len(stacks) != len(agents) or len(stacks) < 2:
            raise ValueError("need matching stack/agent lists of length >= 2")
        if any(int(s) != s or s < 0 for s in stacks):
            raise ValueError("stacks must be nonnegative integers")
        if not 0 < small_blind <= big_blind:
            raise ValueError("invalid blinds")
        self.players = [Player(i, int(s), folded=s == 0) for i, s in enumerate(stacks)]
        self.agents = list(agents)
        self.n = len(stacks)
        self.button = button % self.n
        # Multiway tournaments can have a dead button. Heads-up always has a
        # live button, who posts the small blind and acts first preflop.
        if stacks[self.button] <= 0 and sum(s > 0 for s in stacks) <= 2:
            self.button = self._next(self.button, lambda p: p.stack > 0)
        self.small_blind, self.big_blind = int(small_blind), int(big_blind)
        live_count = sum(s > 0 for s in stacks)
        if blind_seats is None and live_count >= 2:
            sb_position = self.button if live_count == 2 else self._next(self.button, lambda p: p.stack > 0)
            bb_seat = self._next(sb_position, lambda p: p.stack > 0)
            blind_seats = (sb_position, bb_seat)
        self.blind_seats = blind_seats
        if blind_seats is not None:
            sb_position, bb_seat = blind_seats
            if not (0 <= sb_position < self.n and 0 <= bb_seat < self.n and sb_position != bb_seat and stacks[bb_seat] > 0):
                raise ValueError("invalid blind positions")
        self.hand_number = hand_number
        self.rng = rng or random.Random()
        self.deck = list(deck) if deck is not None else list(range(52))
        if deck is None:
            self.rng.shuffle(self.deck)
        if len(set(self.deck)) != len(self.deck) or any(not 0 <= c < 52 for c in self.deck):
            raise ValueError("deck contains duplicate or invalid cards")
        self.deck_index = 0
        self.board: list[int] = []
        self.history: list[dict] = []
        self.observations: list[dict] | None = [] if record_observations else None
        self.street = "preflop"
        self.current_bet = self.big_blind
        self.last_full_raise = self.big_blind
        self.acted_at: list[int | None] = [None] * self.n
        self.payouts = payouts
        self.enforce_time_limit = enforce_time_limit
        self.decision_limit_ns = decision_limit_ns
        self.errors = 0
        self.timeouts = 0
        self.invalid_actions = 0
        self.decisions = 0
        self.max_decision_ns = 0
        self.start_stacks = list(stacks)

    def _next(self, seat, predicate):
        for offset in range(1, self.n + 1):
            other = (seat + offset) % self.n
            if predicate(self.players[other]):
                return other
        raise RuntimeError("no matching seat")

    def _draw(self):
        card = self.deck[self.deck_index]
        self.deck_index += 1
        return card

    def _commit(self, seat, amount):
        p = self.players[seat]
        if amount < 0 or amount > p.stack:
            raise AssertionError("invalid commitment")
        p.stack -= amount
        p.street_bet += amount
        p.committed += amount
        p.all_in = p.stack == 0

    def _eligible(self):
        return [p for p in self.players if not p.folded and p.stack > 0]

    def _call_amount(self, seat):
        p = self.players[seat]
        target = self.current_bet
        if sum(not q.folded and q.stack > 0 for q in self.players) <= 1:
            # A player cannot wager into a dry side pot. A short all-in blind
            # does not force the sole opponent to complete the nominal blind.
            target = max((q.street_bet for q in self.players if not q.folded and q.seat != seat), default=0)
        return min(p.stack, max(0, target - p.street_bet))

    def _min_raise(self):
        # In no-limit there is no "completion" of a short all-in wager.
        # An unacted player raises by at least BB over a short opening bet.
        return self.current_bet + self.last_full_raise

    def _can_raise(self, seat):
        p = self.players[seat]
        previous = self.acted_at[seat]
        # A check is action too: an opening all-in below the minimum bet does
        # not reopen that checked player's raise (Poker TDA 2026, rule 49).
        reopened = previous is None or self.current_bet - previous >= self.last_full_raise
        return (reopened and p.stack + p.street_bet > self.current_bet and
                sum(not q.folded and q.stack > 0 for q in self.players) >= 2)

    def observation(self, seat):
        p = self.players[seat]
        to_call = self._call_amount(seat)
        can_raise = self._can_raise(seat)
        legal = ["fold", "call" if to_call else "check"]
        if can_raise:
            legal.append("raise")
        return {
            "hole_cards": list(p.cards), "board": list(self.board), "seat": seat,
            "button": self.button, "small_blind": self.small_blind,
            "big_blind": self.big_blind, "hand_number": self.hand_number,
            "small_blind_seat": self.blind_seats[0] if self.blind_seats and self.start_stacks[self.blind_seats[0]] else None,
            "big_blind_seat": self.blind_seats[1] if self.blind_seats else None,
            "street": self.street,
            "players": [{"seat": q.seat, "stack": q.stack, "street_bet": q.street_bet,
                         "committed": q.committed, "folded": q.folded, "all_in": q.all_in}
                        for q in self.players],
            "pot": sum(q.committed for q in self.players), "to_call": to_call,
            "current_bet": self.current_bet, "min_raise_to": self._min_raise(),
            "max_raise_to": p.street_bet + p.stack, "can_raise": can_raise,
            "legal_actions": legal, "history": [dict(a) for a in self.history],
            "payouts": list(self.payouts) if self.payouts is not None else None,
        }

    def _get_action(self, seat):
        obs = self.observation(seat)
        if self.observations is not None:
            self.observations.append(obs)
        fallback = {"action": "check" if obs["to_call"] == 0 else "fold"}
        self.decisions += 1
        try:
            agent = self.agents[seat]
            start = perf_counter_ns()
            action = agent(obs) if callable(agent) else agent.act(obs)
            elapsed = perf_counter_ns() - start
            self.max_decision_ns = max(self.max_decision_ns, elapsed)
            if self.enforce_time_limit and elapsed > self.decision_limit_ns:
                self.timeouts += 1
                return fallback
            if not isinstance(action, dict):
                raise ValueError("action must be a dict")
            name = action.get("action")
            if name == "call" and not obs["to_call"]:
                name = "check"
            if name not in obs["legal_actions"]:
                self.invalid_actions += 1
                return fallback
            if name == "raise":
                amount = action.get("amount")
                if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not isfinite(amount):
                    self.invalid_actions += 1
                    return fallback
                amount = int(amount)
                if amount != action["amount"] or amount > obs["max_raise_to"] or amount <= self.current_bet:
                    self.invalid_actions += 1
                    return fallback
                if amount < obs["min_raise_to"] and amount != obs["max_raise_to"]:
                    self.invalid_actions += 1
                    return fallback
                return {"action": name, "amount": amount}
            return {"action": name}
        except Exception:
            self.errors += 1
            return fallback

    def _betting_round(self, first):
        pending = {p.seat for p in self._eligible()}
        seat = (first - 1) % self.n
        while pending:
            live = [p for p in self.players if not p.folded]
            eligible = self._eligible()
            if len(live) <= 1 or not eligible:
                break
            if len(eligible) == 1 and not self._call_amount(eligible[0].seat):
                break
            possible = pending.intersection(p.seat for p in eligible)
            if not possible:
                break
            seat = self._next(seat, lambda p: p.seat in possible)
            p = self.players[seat]
            action = self._get_action(seat)
            name = action["action"]
            before = self.current_bet
            pot_before = sum(q.committed for q in self.players)
            to_call_before = self._call_amount(seat)
            paid = 0
            pending.discard(seat)
            if name == "fold":
                p.folded = True
            elif name == "call":
                paid = to_call_before
                self._commit(seat, paid)
            elif name == "raise":
                target = action["amount"]
                paid = target - p.street_bet
                minimum = self._min_raise()
                self._commit(seat, paid)
                self.current_bet = target
                if target >= minimum:
                    self.last_full_raise = target - before
                    pending = {q.seat for q in self._eligible() if q.seat != seat}
                else:
                    # Short all-ins require a response but do not themselves
                    # reopen raising. Cumulative short raises can reopen it.
                    pending.update(q.seat for q in self._eligible() if q.seat != seat and q.street_bet < target)
            self.acted_at[seat] = self.current_bet
            self.history.append({"seat": seat, "street": self.street, "action": name,
                                 "amount": action["amount"] if name == "raise" else paid,
                                 "paid": paid, "to_call_before": to_call_before,
                                 "current_bet_before": before, "pot_before": pot_before})
        self._return_uncalled()

    def _return_uncalled(self):
        bets = sorted(((p.street_bet, p.seat) for p in self.players), reverse=True)
        if len(bets) >= 2 and bets[0][0] > bets[1][0]:
            excess = bets[0][0] - bets[1][0]
            p = self.players[bets[0][1]]
            p.stack += excess
            p.street_bet -= excess
            p.committed -= excess
            p.all_in = p.stack == 0
            self.history.append({"seat": p.seat, "street": self.street, "action": "return",
                                 "amount": excess, "paid": -excess})

    def _settle(self):
        live = [p for p in self.players if not p.folded]
        if not live:
            raise AssertionError("every player folded")
        if len(live) == 1:
            live[0].stack += sum(p.committed for p in self.players)
            return
        scores = {p.seat: evaluate(list(p.cards) + self.board) for p in live}
        levels = sorted({p.committed for p in self.players if p.committed})
        lower = 0
        for upper in levels:
            contributors = [p for p in self.players if p.committed >= upper]
            pot = (upper - lower) * len(contributors)
            eligible = [p for p in contributors if not p.folded]
            if not eligible:
                raise AssertionError("unreachable pot without an eligible player")
            best = max(scores[p.seat] for p in eligible)
            winners = [p for p in eligible if scores[p.seat] == best]
            winners.sort(key=lambda p: (p.seat - self.button - 1) % self.n)
            share, odd = divmod(pot, len(winners))
            for index, winner in enumerate(winners):
                winner.stack += share + (index < odd)
            lower = upper

    def play(self):
        live = [p for p in self.players if p.stack > 0]
        if len(live) < 2:
            return self.result()
        seat = self.button
        for round_index in range(2):
            for _ in live:
                seat = self._next(seat, lambda p: p.stack > 0)
                p = self.players[seat]
                p.cards = p.cards + (self._draw(),)
        sb_position, bb_seat = self.blind_seats
        sb_seat = sb_position if self.players[sb_position].stack > 0 else None
        for seat, amount in ((sb_seat, self.small_blind), (bb_seat, self.big_blind)):
            if seat is None:
                continue  # dead small blind after the previous big blind busts
            amount = min(amount, self.players[seat].stack)
            self._commit(seat, amount)
            self.history.append({"seat": seat, "street": "preflop", "action": "blind", "amount": amount, "paid": amount})
        self._betting_round(self._next(bb_seat, lambda p: not p.folded))
        for street_index in range(1, 4):
            if sum(not p.folded for p in self.players) < 2:
                break
            self.street = STREETS[street_index]
            for p in self.players:
                p.street_bet = 0
            self.current_bet = 0
            self.last_full_raise = self.big_blind
            self.acted_at = [None] * self.n
            self._draw()  # burn
            self.board.extend(self._draw() for _ in range(3 if street_index == 1 else 1))
            self._betting_round(self._next(self.button, lambda p: not p.folded))
        self._settle()
        if sum(p.stack for p in self.players) != sum(self.start_stacks):
            raise AssertionError("chip conservation failed")
        return self.result()

    def result(self):
        return {"stacks": [p.stack for p in self.players], "board": list(self.board),
                "button": self.button, "small_blind": self.small_blind,
                "big_blind": self.big_blind, "hand_number": self.hand_number,
                "small_blind_position": self.blind_seats[0] if self.blind_seats else None,
                "small_blind_seat": self.blind_seats[0] if self.blind_seats and self.start_stacks[self.blind_seats[0]] else None,
                "big_blind_seat": self.blind_seats[1] if self.blind_seats else None,
                "history": self.history, "errors": self.errors, "timeouts": self.timeouts,
                "invalid_actions": self.invalid_actions, "decisions": self.decisions,
                "max_decision_ns": self.max_decision_ns}


class Tournament:
    """Local sit-and-go. Agent instances belong to this one table only.

    blind_schedule is a sequence of (small_blind, big_blind) pairs, advanced
    every hands_per_level hands. The default schedule doubles; it is NOT an
    assertion about the unreleased official blind schedule.
    """

    def __init__(self, agents: Sequence[Callable], *, seed=0, stacks=None,
                 initial_stack=2000, blind_schedule=None, hands_per_level=25,
                 max_hands=200, payouts=None, enforce_time_limit=False,
                 retain_history=False, initial_button=None):
        self.agents = list(agents)
        self.n = len(agents)
        if not 2 <= self.n <= 8:
            raise ValueError("local engine supports 2..8 players")
        self.stacks = list(stacks) if stacks is not None else [initial_stack] * self.n
        if len(self.stacks) != self.n or any(s <= 0 or int(s) != s for s in self.stacks):
            raise ValueError("tournament starting stacks must be positive integers")
        if max_hands < 1 or hands_per_level < 1:
            raise ValueError("invalid tournament duration")
        self.rng = random.Random(seed)
        self.blind_schedule = list(blind_schedule) if blind_schedule is not None else [(10 * 2 ** i, 20 * 2 ** i) for i in range(8)]
        if not self.blind_schedule or any(not 0 < sb <= bb for sb, bb in self.blind_schedule):
            raise ValueError("invalid blind schedule")
        self.hands_per_level = hands_per_level
        self.max_hands = max_hands
        if payouts is not None and (len(payouts) != self.n or any(not isfinite(float(v)) for v in payouts)):
            raise ValueError("provide exactly one finite payout per seat")
        self.payouts = list(payouts) if payouts is not None else None
        self.enforce_time_limit = enforce_time_limit
        self.retain_history = retain_history
        self.button = self.rng.randrange(self.n) if initial_button is None else initial_button % self.n
        self.sb_position = self.button if self.n == 2 else (self.button + 1) % self.n
        self.bb_seat = (self.sb_position + 1) % self.n

    def run(self):
        ranks: list[float | None] = [None] * self.n
        points = [0.0] * self.n if self.payouts is not None else None
        history = []
        totals = {"errors": 0, "timeouts": 0, "invalid_actions": 0, "decisions": 0, "max_decision_ns": 0}
        hands = 0
        for hand_number in range(self.max_hands):
            alive = sum(s > 0 for s in self.stacks)
            if alive <= 1:
                break
            sb, bb = self.blind_schedule[min(hand_number // self.hands_per_level, len(self.blind_schedule) - 1)]
            hand = Hand(self.stacks, self.agents, button=self.button, small_blind=sb, big_blind=bb,
                        hand_number=hand_number, rng=self.rng, payouts=self.payouts,
                        enforce_time_limit=self.enforce_time_limit,
                        blind_seats=(self.sb_position, self.bb_seat))
            before = self.stacks[:]
            result = hand.play()
            self.stacks = result["stacks"]
            hands += 1
            for key in ("errors", "timeouts", "invalid_actions", "decisions"):
                totals[key] += result[key]
            totals["max_decision_ns"] = max(totals["max_decision_ns"], result["max_decision_ns"])
            if self.retain_history:
                history.append(result)
            eliminated = [i for i, s in enumerate(self.stacks) if not s and before[i]]
            # Simultaneous busts: larger beginning-of-hand stacks finish ahead;
            # equal beginning stacks tie and share the occupied payout places.
            place = alive
            for stack in sorted({before[i] for i in eliminated}):
                tied = [i for i in eliminated if before[i] == stack]
                average_rank = (place + place - len(tied) + 1) / 2
                value = sum(self.payouts[place - len(tied):place]) / len(tied) if self.payouts is not None else None
                for i in tied:
                    ranks[i] = average_rank
                    if points is not None:
                        points[i] = value
                place -= len(tied)
            # Advance BB to the next surviving seat. Under the tournament dead
            # button rule, the new SB position is the old BB (dead if busted),
            # and the new button is the old SB position. Heads-up overrides
            # dead positions so no survivor posts BB twice consecutively.
            old_bb = self.bb_seat
            new_bb = next((old_bb + offset) % self.n for offset in range(1, self.n + 1) if self.stacks[(old_bb + offset) % self.n])
            remaining = sum(s > 0 for s in self.stacks)
            if remaining <= 2:
                self.button = next((new_bb - offset) % self.n for offset in range(1, self.n + 1) if self.stacks[(new_bb - offset) % self.n])
                self.sb_position = self.button
            else:
                self.button = self.sb_position
                self.sb_position = old_bb
            self.bb_seat = new_bb
        live = [i for i, s in enumerate(self.stacks) if s]
        if len(live) == 1:
            ranks[live[0]] = 1.0
        if points is not None:
            remaining = icm_values(self.stacks, self.payouts[:len(live)])
            for i in live:
                points[i] = remaining[i]
        expected_ranks = icm_values(self.stacks, list(range(1, len(live) + 1)))
        for i in range(self.n):
            if ranks[i] is not None:
                expected_ranks[i] = ranks[i]
        return {"stacks": self.stacks[:], "ranks": ranks, "expected_ranks": expected_ranks,
                "points": points, "hands": hands, "history": history, **totals}


def simulate_tournament(agents, **kwargs):
    return Tournament(agents, **kwargs).run()
