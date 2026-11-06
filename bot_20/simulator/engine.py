"""Integer-chip no-limit Hold'em with audited local observation semantics.

This is an independently authored testing environment, not the unavailable
contest starter kit. Explicit assumptions are documented in simulator/README.md.
"""

from dataclasses import dataclass, field
import random
from time import perf_counter_ns
from uuid import uuid4

from pokerbot.evaluator import evaluate
from pokerbot.icm import icm_equities
from pokerbot.interface import is_legal, passive


class IllegalAction(ValueError):
    pass


@dataclass
class DecisionMetrics:
    decisions: int = 0
    errors: int = 0
    illegal: int = 0
    over_1ms: int = 0
    fallbacks: int = 0
    elapsed_ns: int = 0
    max_ns: int = 0
    timings_ns: list = field(default_factory=list)

    def add(self, other):
        for key in ("decisions", "errors", "illegal", "over_1ms", "fallbacks", "elapsed_ns"):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        self.max_ns = max(self.max_ns, other.max_ns)
        self.timings_ns.extend(other.timings_ns)

    def summary(self):
        data = {key: getattr(self, key) for key in
                ("decisions", "errors", "illegal", "over_1ms", "fallbacks", "elapsed_ns", "max_ns")}
        data["mean_us"] = self.elapsed_ns / max(1, self.decisions) / 1000
        if self.timings_ns:
            values = sorted(self.timings_ns)
            for label, fraction in (("p50_us", .5), ("p95_us", .95), ("p99_us", .99), ("p999_us", .999)):
                data[label] = values[min(len(values) - 1, int(len(values) * fraction))] / 1000
        return data


class Hand:
    """One hand. ``actor`` is a seat index, or None when ``done`` is true.

    Action 'raise' always specifies a final street total. No special 'allin'
    action exists: short calls and raises to max_raise_to express all-ins.
    """

    def __init__(self, stacks, button=0, small_blind=10, big_blind=20,
                 seed=0, deck=None, hand_id=0, payouts=None,
                 hands_remaining=200, table_id=0):
        self.n = len(stacks)
        if not 2 <= self.n <= 8 or any(type(s) is not int or s < 0 for s in stacks):
            raise ValueError("require 2..8 nonnegative integer starting stacks")
        if sum(s > 0 for s in stacks) < 2:
            raise ValueError("a hand requires two live stacks")
        if type(small_blind) is not int or type(big_blind) is not int or not 0 < small_blind <= big_blind:
            raise ValueError("require positive integer small_blind <= big_blind")
        if not 0 <= button < self.n:
            raise ValueError("button out of range")
        self.hand_start_stacks = list(stacks)
        self.stacks = list(stacks)
        self.total_chips = sum(stacks)
        self.folded = [s == 0 for s in stacks]
        self.committed = [0] * self.n
        self.street_committed = [0] * self.n
        self.last_acted_at = [None] * self.n
        self.button = button if stacks[button] else self._next(button, lambda i: stacks[i] > 0)
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.hand_id = hand_id
        self.table_id = table_id
        self.payouts = list(payouts) if payouts is not None else None
        self.hands_remaining = hands_remaining
        self.street = 0
        self.board = []
        self.holes = [[] for _ in stacks]
        self.history = []
        self.refunds = [0] * self.n
        self.pots = []
        self.awards = [0] * self.n
        self.done = False
        self.actor = None
        self.last_full_raise = big_blind
        self.current_bet = big_blind
        if deck is None:
            self.deck = list(range(52))
            random.Random(seed).shuffle(self.deck)
        else:
            self.deck = list(deck)
            if len(self.deck) != 52 or set(self.deck) != set(range(52)):
                raise ValueError("deck must be a permutation of 52 cards")
        self.deck_index = 0
        seats = self._ordered(self.button, lambda i: stacks[i] > 0)
        for _ in range(2):
            for i in seats:
                self.holes[i].append(self._draw())
        if len(seats) == 2:
            self.sb_seat = self.button
            self.bb_seat = self._next(self.button, lambda i: stacks[i] > 0)
        else:
            self.sb_seat = seats[0]
            self.bb_seat = seats[1]
        self._pay(self.sb_seat, min(small_blind, self.stacks[self.sb_seat]))
        self._pay(self.bb_seat, min(big_blind, self.stacks[self.bb_seat]))
        self._progress(self.bb_seat)

    def _next(self, after, condition):
        for step in range(1, self.n + 1):
            seat = (after + step) % self.n
            if condition(seat):
                return seat
        return None

    def _ordered(self, after, condition):
        return [(after + step) % self.n for step in range(1, self.n + 1)
                if condition((after + step) % self.n)]

    def _draw(self):
        card = self.deck[self.deck_index]
        self.deck_index += 1
        return card

    def _pay(self, seat, amount):
        self.stacks[seat] -= amount
        self.committed[seat] += amount
        self.street_committed[seat] += amount

    @property
    def pot(self):
        return 0 if self.done else sum(self.committed)

    def _can_raise(self, seat):
        if self.stacks[seat] + self.street_committed[seat] <= self.current_bet:
            return False
        if not any(i != seat and not self.folded[i] and self.stacks[i] > 0 for i in range(self.n)):
            return False
        previous = self.last_acted_at[seat]
        return previous is None or self.current_bet - previous >= self.last_full_raise

    def observation(self):
        if self.done or self.actor is None:
            raise ValueError("completed hand has no decision observation")
        seat = self.actor
        return {
            "hero": seat, "hand_id": self.hand_id, "street": self.street,
            "button": self.button, "sb_seat": self.sb_seat, "bb_seat": self.bb_seat,
            "small_blind": self.small_blind, "big_blind": self.big_blind,
            "stacks": self.stacks.copy(), "hand_start_stacks": self.hand_start_stacks.copy(),
            "committed": self.committed.copy(), "street_committed": self.street_committed.copy(),
            "folded": self.folded.copy(),
            "all_in": [not self.folded[i] and self.stacks[i] == 0 for i in range(self.n)],
            "hole_cards": self.holes[seat].copy(), "board": self.board.copy(),
            "pot": self.pot, "to_call": max(0, self.current_bet - self.street_committed[seat]),
            "min_raise_to": self.current_bet + self.last_full_raise,
            "max_raise_to": self.street_committed[seat] + self.stacks[seat],
            "can_raise": self._can_raise(seat),
            "active_players": sum(not f for f in self.folded),
            "history": [event.copy() for event in self.history],
            "payouts": self.payouts.copy() if self.payouts is not None else None,
            "hands_remaining": self.hands_remaining, "table_id": self.table_id,
        }

    def act(self, action):
        obs = self.observation()
        if not is_legal(obs, action):
            raise IllegalAction(f"illegal action {action!r} at seat {self.actor}, street {self.street}: "
                                f"to_call={obs['to_call']}, raise={obs['can_raise']} "
                                f"bounds={obs['min_raise_to']}..{obs['max_raise_to']}")
        seat = self.actor
        kind = action["action"]
        amount = action.get("amount", 0)
        event = {"seat": seat, "street": self.street, "action": kind,
                 "amount": amount, "to_call": obs["to_call"], "pot_before": self.pot}
        self.history.append(event)
        if kind == "fold":
            self.folded[seat] = True
        elif kind == "call":
            self._pay(seat, amount)
        elif kind == "raise":
            increment = amount - self.current_bet
            self._pay(seat, amount - self.street_committed[seat])
            self.current_bet = amount
            if increment >= self.last_full_raise:
                self.last_full_raise = increment
        self.last_acted_at[seat] = self.current_bet
        self.actor = None
        self._progress(seat)
        self.assert_invariants()

    def _refund_uncalled(self):
        ordered = sorted(range(self.n), key=self.street_committed.__getitem__, reverse=True)
        first, second = ordered[:2]
        excess = self.street_committed[first] - self.street_committed[second]
        if excess:
            self.street_committed[first] -= excess
            self.committed[first] -= excess
            self.stacks[first] += excess
            self.refunds[first] += excess

    def _deal_street(self):
        self.street += 1
        self._draw()  # Burn; not revealed to any policy.
        self.board.extend(self._draw() for _ in range(3 if self.street == 1 else 1))
        self.street_committed = [0] * self.n
        self.last_acted_at = [None] * self.n
        self.current_bet = 0
        self.last_full_raise = self.big_blind

    def _progress(self, after):
        while not self.done:
            live = [i for i in range(self.n) if not self.folded[i]]
            if len(live) == 1:
                self._refund_uncalled()
                amount = sum(self.committed)
                self.awards[live[0]] += amount
                self.stacks[live[0]] += amount
                self.pots.append({"amount": amount, "eligible": live, "winners": live})
                self.done = True
                self.actor = None
                return
            can_act = [i for i in live if self.stacks[i] > 0]
            if len(can_act) <= 1:
                # No side betting into all-ins. A short blind cannot create an
                # artificial expensive call when nobody can contest the excess.
                actual_bet = max(self.street_committed[i] for i in live)
                if can_act and self.street_committed[can_act[0]] < actual_bet:
                    self.current_bet = actual_bet
                    self.actor = can_act[0]
                    return
                self._refund_uncalled()
                while self.street < 3:
                    self._deal_street()
                self._showdown()
                return
            needs_action = {i for i in can_act
                            if self.last_acted_at[i] is None or self.street_committed[i] < self.current_bet}
            if needs_action:
                self.actor = self._next(after, needs_action.__contains__)
                return
            self._refund_uncalled()
            if self.street == 3:
                self._showdown()
                return
            self._deal_street()
            after = self.button

    def _showdown(self):
        ranks = {i: evaluate(self.holes[i] + self.board) for i in range(self.n) if not self.folded[i]}
        previous = 0
        for level in sorted(set(c for c in self.committed if c > 0)):
            contributors = [i for i, c in enumerate(self.committed) if c >= level]
            amount = (level - previous) * len(contributors)
            eligible = [i for i in contributors if not self.folded[i]]
            if not eligible:
                raise AssertionError("side pot has no live eligible player")
            best = max(ranks[i] for i in eligible)
            winners = [i for i in eligible if ranks[i] == best]
            share, odd = divmod(amount, len(winners))
            for i in winners:
                self.awards[i] += share
                self.stacks[i] += share
            for i in self._ordered(self.button, winners.__contains__)[:odd]:
                self.awards[i] += 1
                self.stacks[i] += 1
            self.pots.append({"amount": amount, "eligible": eligible, "winners": winners})
            previous = level
        self.done = True
        self.actor = None

    def assert_invariants(self):
        if any(type(s) is not int or s < 0 for s in self.stacks):
            raise AssertionError("negative or noninteger stack")
        if sum(self.stacks) + self.pot != self.total_chips:
            raise AssertionError("chips were created or destroyed")
        if any(s < 0 or s > c for s, c in zip(self.street_committed, self.committed)):
            raise AssertionError("invalid commitment accounting")
        if not self.done and (self.actor is None or self.folded[self.actor] or self.stacks[self.actor] == 0):
            raise AssertionError("invalid next actor")

    def result(self):
        if not self.done:
            raise ValueError("hand not complete")
        return {"stacks": self.stacks.copy(), "awards": self.awards.copy(),
                "contributions": self.committed.copy(), "refunds": self.refunds.copy(),
                "pots": self.pots.copy(), "board": self.board.copy(),
                "holes": [h.copy() for h in self.holes], "history": self.history.copy()}

    def next_button(self):
        """Live-button progression, with standard heads-up transition repair."""
        if not self.done:
            raise ValueError("finish hand before advancing button")
        live = [i for i, chips in enumerate(self.stacks) if chips > 0]
        if len(live) < 2:
            return None
        if len(live) == 2:
            # The next live player after the previous BB takes the next BB.
            # The other survivor is button/SB, avoiding consecutive big blinds
            # when a three-handed button busts on the transition to heads-up.
            next_bb = self._next(self.bb_seat, live.__contains__)
            return self._next(next_bb, live.__contains__)
        return self._next(self.button, live.__contains__)


def _policy_function(policy):
    if hasattr(policy, "act"):
        return policy.act
    if hasattr(policy, "decide"):
        return policy.decide
    if callable(policy):
        return policy
    raise TypeError("policy must define act/decide or be callable")


def play_hand(hand, policies, strict=True, enforce_time=False, timing=False, on_observation=None):
    """Play a hand; errors/timeouts optionally use contest-described check/fold.

    Decision timing excludes observation creation. Wall-clock OS preemption is
    included, matching a wall-time interpretation. No latency claim is made for
    the simulator itself. ``strict`` raises errors/illegal actions immediately.
    """
    if len(policies) != hand.n:
        raise ValueError("one policy per seat required")
    functions = [_policy_function(p) for p in policies]
    metrics = [DecisionMetrics() for _ in policies]
    while not hand.done:
        seat = hand.actor
        obs = hand.observation()
        if on_observation:
            on_observation(obs)
        measure = metrics[seat]
        measure.decisions += 1
        action = None
        failed = False
        start = perf_counter_ns()
        try:
            action = functions[seat](obs)
        except Exception:
            measure.errors += 1
            if strict:
                raise
            failed = True
        elapsed = perf_counter_ns() - start
        measure.elapsed_ns += elapsed
        measure.max_ns = max(measure.max_ns, elapsed)
        if timing:
            measure.timings_ns.append(elapsed)
        if elapsed > 1_000_000:
            measure.over_1ms += 1
            if enforce_time:
                failed = True
        # Compare against a fresh observation; a policy mutating its own input
        # cannot change the engine's legal bounds or visible state.
        if not failed and not is_legal(hand.observation(), action):
            measure.illegal += 1
            if strict:
                raise IllegalAction(f"policy at seat {seat} returned {action!r}")
            failed = True
        if failed:
            measure.fallbacks += 1
            action = passive(hand.observation())
        hand.act(action)
    hand.assert_invariants()
    return metrics


def play_tournament(policies, *, seed=0, starting_stack=2000, max_hands=200,
                    blind_levels=None, level_hands=25, payouts=None, button=None,
                    strict=True, enforce_time=False, timing=False,
                    table_id=None, on_observation=None, retain_hands=False):
    """Run one LOCAL sit-and-go with explicit configurable assumptions.

    Remaining points at the cutoff are assigned by exact ICM. Simultaneous
    eliminations are ordered by hand-start chips; equal stacks split the awards
    for their tied positions. Policies are reset once, before the table.
    """
    n = len(policies)
    if not 2 <= n <= 8 or type(starting_stack) is not int or starting_stack <= 0:
        raise ValueError("require 2..8 policies and positive starting chips")
    if blind_levels is None:
        raise ValueError("supply an explicit local blind schedule; official schedule unavailable")
    if not blind_levels or level_hands < 1 or max_hands < 1:
        raise ValueError("invalid blind schedule or hand cap")
    if payouts is None:
        raise ValueError("supply explicit local payouts; official payouts unavailable")
    if len(payouts) != n or any(p < 0 for p in payouts) or any(a < b for a, b in zip(payouts, payouts[1:])):
        raise ValueError("payouts must be nonnegative nonincreasing and match seats")
    for sb, bb in blind_levels:
        if type(sb) is not int or type(bb) is not int or not 0 < sb <= bb:
            raise ValueError("invalid blind level")
    for policy in policies:
        if hasattr(policy, "reset"):
            policy.reset()
    rng = random.Random(seed)
    # Never expose the reproducible deck seed through an observation tag.
    # The policy may compare this public opaque identifier only for table reset.
    public_table_id = uuid4().hex if table_id is None else table_id
    if button is None:
        button = rng.randrange(n)
    stacks = [starting_stack] * n
    scores = [0.0] * n
    metrics = [DecisionMetrics() for _ in policies]
    records = []
    hands_played = 0
    alive = list(range(n))
    for hand_id in range(max_hands):
        sb, bb = blind_levels[min(hand_id // level_hands, len(blind_levels) - 1)]
        deck = list(range(52))
        rng.shuffle(deck)
        before = stacks.copy()
        hand = Hand(stacks, button=button, small_blind=sb, big_blind=bb, deck=deck,
                    hand_id=hand_id, payouts=payouts, hands_remaining=max_hands - hand_id,
                    table_id=public_table_id)
        hand_metrics = play_hand(hand, policies, strict=strict, enforce_time=enforce_time,
                                 timing=timing, on_observation=on_observation)
        for measure, extra in zip(metrics, hand_metrics):
            measure.add(extra)
        stacks = hand.stacks.copy()
        hands_played += 1
        if retain_hands:
            records.append(hand.result())
        eliminated = sorted((i for i in alive if stacks[i] == 0), key=lambda i: before[i])
        remaining_count = len(alive)
        j = 0
        while j < len(eliminated):
            end = j + 1
            while end < len(eliminated) and before[eliminated[end]] == before[eliminated[j]]:
                end += 1
            tied = eliminated[j:end]
            award = sum(payouts[remaining_count - len(tied):remaining_count]) / len(tied)
            for seat in tied:
                scores[seat] = award
            remaining_count -= len(tied)
            j = end
        alive = [i for i in alive if stacks[i] > 0]
        if len(alive) == 1:
            break
        # Local convention: moving/live button. Empty/eliminated seats skipped.
        button = hand.next_button()
    residual = icm_equities([stacks[i] for i in alive], list(payouts[:len(alive)]))
    for seat, award in zip(alive, residual):
        scores[seat] = award
    if abs(sum(scores) - sum(payouts)) > 1e-8:
        raise AssertionError("tournament points not conserved")
    return {"scores": scores, "stacks": stacks, "hands": hands_played,
            "survivors": alive, "metrics": [m.summary() for m in metrics],
            "winner": alive[0] if len(alive) == 1 else None,
            "cutoff": len(alive) > 1, "hand_records": records}
