"""No-limit hold'em sit-and-go engine.

Conventions (chosen to match the obvious reading of the contest rules and the
repository's other local simulator so bots can be exchanged):

* Raise amounts are the TOTAL chips committed on the current street.
* Observation stacks are chips behind, i.e. after posting blinds/bets.
* Moving button: the big blind advances to the next live seat each hand; the
  small blind and button are the previous live seats (heads-up: button = SB).
* Minimum raise follows TDA rules: a raise must be at least the last full raise
  increment; an all-in that does not reach a full raise does not reopen action
  for players who already acted.
* Eliminated players finish by the order of elimination; simultaneous
  eliminations are ranked by their starting stack this hand (ties split
  points). At the hand cap, survivors split the remaining payouts by ICM.
"""

from dataclasses import dataclass, field
from random import Random
from time import perf_counter_ns
from typing import Callable, Sequence

from .cards import evaluate
from .icm import icm

PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3
STREET_NAMES = ("preflop", "flop", "turn", "river")

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"


@dataclass(frozen=True, slots=True)
class Action:
    kind: str
    amount: int = 0  # total street commitment for RAISE

    def __repr__(self):
        return f"{self.kind}" + (f"({self.amount})" if self.kind == RAISE else "")


@dataclass(frozen=True, slots=True)
class Event:
    seat: int
    street: int
    kind: str
    amount: int  # resulting street commitment (0 for fold / check)
    paid: int = 0  # chips paid by this event


@dataclass(slots=True)
class Observation:
    """Everything a bot may see. Mutable only by the engine; bots must not mutate."""
    seat: int
    hole: tuple[int, int]
    board: tuple[int, ...]
    n_seats: int
    stacks: tuple[int, ...]          # chips behind
    street_bets: tuple[int, ...]     # committed this street
    contributions: tuple[int, ...]   # committed this hand
    folded: tuple[bool, ...]
    all_in: tuple[bool, ...]
    active: tuple[bool, ...]         # dealt into this hand
    starting_stacks: tuple[int, ...]  # beginning of hand (before blinds)
    button: int
    sb_seat: int
    bb_seat: int
    sb: int
    bb: int
    ante: int
    pot: int
    to_call: int
    min_raise_to: int
    max_raise_to: int
    can_raise: bool
    street: int
    hand_number: int
    hands_left: int
    history: tuple[Event, ...]
    payouts: tuple[float, ...]

    @property
    def players_in_hand(self) -> int:
        return sum(1 for i in range(self.n_seats) if self.active[i] and not self.folded[i])

    @property
    def players_alive(self) -> int:
        return sum(1 for i in range(self.n_seats) if self.active[i])


class Bot:
    """Interface. reset() is called once per table; act() once per decision."""

    def reset(self, seed: int, seat: int, n_seats: int) -> None:  # pragma: no cover
        pass

    def act(self, obs: Observation) -> Action:  # pragma: no cover
        raise NotImplementedError


BotFactory = Callable[[], Bot]

DEFAULT_SCHEDULE = (
    (10, 20, 0), (20, 40, 0), (40, 80, 0), (80, 160, 0),
    (160, 320, 0), (320, 640, 0), (640, 1280, 0), (1280, 2560, 0),
)


@dataclass(frozen=True, slots=True)
class Config:
    n_seats: int = 8
    starting_stack: int = 2000
    max_hands: int = 200
    hands_per_level: int = 25
    schedule: tuple[tuple[int, int, int], ...] = DEFAULT_SCHEDULE  # (sb, bb, ante)
    payouts: tuple[float, ...] = (0.5, 0.3, 0.2, 0, 0, 0, 0, 0)
    decision_limit_ns: int = 1_000_000
    strict_timing: bool = False  # replace late actions with the fallback
    record_hands: bool = False

    def blinds(self, hand_number: int) -> tuple[int, int, int]:
        level = min(hand_number // self.hands_per_level, len(self.schedule) - 1)
        return self.schedule[level]


@dataclass(slots=True)
class HandRecord:
    hand_number: int
    button: int
    stacks_before: tuple[int, ...]
    stacks_after: tuple[int, ...]
    hole: tuple[tuple[int, ...], ...]
    board: tuple[int, ...]
    events: tuple[Event, ...]


@dataclass(slots=True)
class TournamentResult:
    points: list[float]
    final_stacks: list[int]
    hands: int
    placements: list[float]      # average finishing place (1 = winner)
    decisions: list[int]
    timeouts: list[int]
    errors: list[int]
    illegal: list[int]
    max_latency_ns: list[int]
    total_latency_ns: list[int]
    hand_records: list[HandRecord] = field(default_factory=list)


class Table:
    """Mutable table state for one tournament."""

    def __init__(self, bots: Sequence[Bot], config: Config, seed: int):
        self.bots = list(bots)
        self.cfg = config
        self.n = config.n_seats
        assert len(self.bots) == self.n
        self.rng = Random(seed)
        self.seed = seed
        self.stacks = [config.starting_stack] * self.n
        self.hand_number = 0
        self.bb_seat = self.rng.randrange(self.n) if self.n > 2 else 1
        self.finished = []  # list of lists of seats eliminated, chronological
        self.decisions = [0] * self.n
        self.timeouts = [0] * self.n
        self.errors = [0] * self.n
        self.illegal = [0] * self.n
        self.max_latency = [0] * self.n
        self.total_latency = [0] * self.n
        self.records = []
        for seat, bot in enumerate(self.bots):
            bot.reset(seed * 1000003 + seat, seat, self.n)

    # ---------------------------------------------------------------
    def _next_live(self, seat: int) -> int:
        for k in range(1, self.n + 1):
            s = (seat + k) % self.n
            if self.stacks[s] > 0:
                return s
        raise RuntimeError("no live seat")

    def _prev_live(self, seat: int) -> int:
        for k in range(1, self.n + 1):
            s = (seat - k) % self.n
            if self.stacks[s] > 0:
                return s
        raise RuntimeError("no live seat")

    def alive(self) -> list[int]:
        return [i for i in range(self.n) if self.stacks[i] > 0]

    def play_hand(self) -> None:
        n = self.n
        cfg = self.cfg
        stacks = self.stacks
        alive = self.alive()
        starting = tuple(stacks)
        sb, bb, ante = cfg.blinds(self.hand_number)
        # seats
        bb_seat = self._next_live(self.bb_seat)
        if len(alive) == 2:
            sb_seat = self._prev_live(bb_seat)
            button = sb_seat
        else:
            sb_seat = self._prev_live(bb_seat)
            button = self._prev_live(sb_seat)
        self.bb_seat = bb_seat
        active = tuple(s > 0 for s in stacks)
        folded = [not a for a in active]
        street_bets = [0] * n
        contrib = [0] * n
        history: list[Event] = []
        payouts = cfg.payouts
        hands_left = cfg.max_hands - self.hand_number - 1

        deck = list(range(52))
        self.rng.shuffle(deck)
        pos = 0
        hole = [() for _ in range(n)]
        for s in alive:
            hole[s] = (deck[pos], deck[pos + 1])
            pos += 2
        board_full = tuple(deck[pos:pos + 5])

        def pay(seat, amount):
            amount = min(stacks[seat], amount)
            stacks[seat] -= amount
            street_bets[seat] += amount
            contrib[seat] += amount
            return amount

        if ante:
            for s in alive:
                a = min(stacks[s], ante)
                stacks[s] -= a
                contrib[s] += a  # antes are not street bets
        pay(sb_seat, sb)
        pay(bb_seat, bb)

        board_len = 0
        for street in (PREFLOP, FLOP, TURN, RIVER):
            in_hand = [i for i in alive if not folded[i]]
            if len(in_hand) <= 1:
                break
            with_chips = [i for i in in_hand if stacks[i] > 0]
            if street != PREFLOP:
                street_bets = [0] * n
                if len(with_chips) <= 1:
                    # everybody (or everybody but one) is all-in: run out the board.
                    board_len = (0, 3, 4, 5)[street]
                    continue
            board_len = (0, 3, 4, 5)[street]
            board = board_full[:board_len]
            current_bet = bb if street == PREFLOP else 0
            last_full_raise = bb
            acted_to = [None] * n  # street bet level this seat already acted at
            pending = [i for i in in_hand if stacks[i] > 0]
            if street == PREFLOP:
                # action starts left of the BB; heads-up: SB (button) first
                actor = bb_seat
            else:
                actor = button
            # Preflop, the blinds may still act even though they posted.
            while pending and sum(1 for i in in_hand if not folded[i]) > 1:
                # next pending seat after actor
                nxt = None
                for k in range(1, n + 1):
                    s = (actor + k) % n
                    if s in pending:
                        nxt = s
                        break
                if nxt is None:
                    break
                seat = nxt
                actor = seat
                pending.remove(seat)
                if folded[seat] or stacks[seat] == 0:
                    continue
                # If every other live player is all-in or folded, this seat only
                # needs to match the largest live commitment.
                others_with_chips = [i for i in in_hand if i != seat and not folded[i] and stacks[i] > 0]
                live_max = max(street_bets[i] for i in in_hand if i != seat and not folded[i])
                if not others_with_chips and street_bets[seat] >= live_max:
                    continue
                # A nominal full blind is not callable when all opponents are
                # already all-in below it. Do not quote inaccessible excess.
                to_call = max(0, (current_bet if others_with_chips else live_max) - street_bets[seat])
                can_raise = (
                    bool(others_with_chips)
                    and (acted_to[seat] is None or current_bet - acted_to[seat] >= last_full_raise)
                    and stacks[seat] + street_bets[seat] > current_bet
                )
                max_raise_to = stacks[seat] + street_bets[seat]
                min_raise_to = min(current_bet + last_full_raise, max_raise_to)
                obs = Observation(
                    seat=seat, hole=hole[seat], board=board, n_seats=n,
                    stacks=tuple(stacks), street_bets=tuple(street_bets), contributions=tuple(contrib),
                    folded=tuple(folded),
                    all_in=tuple(active[i] and not folded[i] and stacks[i] == 0 for i in range(n)),
                    active=active, starting_stacks=starting, button=button, sb_seat=sb_seat,
                    bb_seat=bb_seat, sb=sb, bb=bb, ante=ante, pot=sum(contrib), to_call=to_call,
                    min_raise_to=min_raise_to, max_raise_to=max_raise_to, can_raise=can_raise,
                    street=street, hand_number=self.hand_number, hands_left=hands_left,
                    history=tuple(history), payouts=payouts,
                )
                fallback = Action(CHECK) if to_call == 0 else Action(FOLD)
                t0 = perf_counter_ns()
                try:
                    action = self.bots[seat].act(obs)
                    failed = False
                except Exception:
                    action = fallback
                    failed = True
                dt = perf_counter_ns() - t0
                self.decisions[seat] += 1
                self.total_latency[seat] += dt
                if dt > self.max_latency[seat]:
                    self.max_latency[seat] = dt
                if failed:
                    self.errors[seat] += 1
                if dt > cfg.decision_limit_ns:
                    self.timeouts[seat] += 1
                    if cfg.strict_timing:
                        action = fallback
                # legality
                kind = action.kind if isinstance(action, Action) else None
                ok = True
                if kind == CHECK:
                    ok = to_call == 0
                elif kind == CALL:
                    ok = to_call > 0
                elif kind == RAISE:
                    amt = action.amount
                    ok = (can_raise and isinstance(amt, int) and current_bet < amt <= max_raise_to
                          and (amt >= min_raise_to or amt == max_raise_to))
                elif kind != FOLD:
                    ok = False
                if not ok:
                    self.illegal[seat] += 1
                    action = fallback
                    kind = action.kind
                if kind == FOLD:
                    folded[seat] = True
                    history.append(Event(seat, street, FOLD, 0))
                elif kind == CHECK:
                    history.append(Event(seat, street, CHECK, street_bets[seat]))
                elif kind == CALL:
                    paid = pay(seat, to_call)
                    history.append(Event(seat, street, CALL, street_bets[seat], paid))
                else:
                    amt = action.amount
                    inc = amt - current_bet
                    paid = pay(seat, amt - street_bets[seat])
                    if inc >= last_full_raise:
                        last_full_raise = inc
                    current_bet = amt
                    for i in in_hand:
                        if i != seat and not folded[i] and stacks[i] > 0 and street_bets[i] < current_bet and i not in pending:
                            pending.append(i)
                    history.append(Event(seat, street, RAISE, amt, paid))
                acted_to[seat] = current_bet

        # showdown / settlement
        in_hand = [i for i in alive if not folded[i]]
        total = sum(contrib)
        if len(in_hand) == 1:
            stacks[in_hand[0]] += total
            board_shown = board_full[:board_len]
        else:
            board_shown = board_full
            scores = {i: evaluate(hole[i] + board_full) for i in in_hand}
            self._settle(contrib, folded, scores, button, alive)
        assert sum(stacks) == sum(starting), "chip conservation"
        if cfg.record_hands:
            self.records.append(HandRecord(self.hand_number, button, starting, tuple(stacks),
                                           tuple(hole), board_shown, tuple(history)))
        # eliminations
        busted = [i for i in alive if stacks[i] == 0]
        if busted:
            # rank by starting stack: bigger starting stack finishes higher.
            busted.sort(key=lambda i: starting[i])
            groups = []
            for i in busted:
                if groups and starting[groups[-1][0]] == starting[i]:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            self.finished.extend(groups)
        self.hand_number += 1

    def _settle(self, contrib, folded, scores, button, alive):
        n = self.n
        stacks = self.stacks
        contrib = list(contrib)
        # refund uncalled excess of the largest contributor
        order = sorted(range(n), key=lambda i: contrib[i], reverse=True)
        top, second = order[0], order[1]
        if contrib[top] > contrib[second]:
            refund = contrib[top] - contrib[second]
            stacks[top] += refund
            contrib[top] -= refund
        levels = sorted({contrib[i] for i in range(n) if contrib[i] > 0 and not folded[i]})
        prev = 0
        distributed = 0
        top_winner = None
        for level in levels:
            pot = 0
            for i in range(n):
                pot += max(0, min(contrib[i], level) - prev)
            eligible = [i for i in alive if not folded[i] and contrib[i] >= level]
            best = max(scores[i] for i in eligible)
            winners = [i for i in eligible if scores[i] == best]
            share, odd = divmod(pot, len(winners))
            distributed += pot
            top_winner = winners[0]
            for w in winners:
                stacks[w] += share
            # odd chips to the first winner clockwise from the button
            k = 1
            while odd:
                s = (button + k) % n
                if s in winners:
                    stacks[s] += 1
                    odd -= 1
                k += 1
            prev = level
        # chips of folded players above the top live level (possible only with antes
        # that exceed short stacks): the last live claimant takes them.
        leftover = sum(contrib) - distributed
        if leftover > 0 and top_winner is not None:
            stacks[top_winner] += leftover

    def run(self) -> TournamentResult:
        cfg = self.cfg
        while self.hand_number < cfg.max_hands and len(self.alive()) > 1:
            self.play_hand()
        return self._score()

    def _score(self) -> TournamentResult:
        n = self.n
        payouts = list(self.cfg.payouts) + [0.0] * n
        points = [0.0] * n
        places = [0.0] * n
        # finished groups: chronological; place numbering from the bottom.
        place = n  # worst place
        for group in self.finished:
            k = len(group)
            # group occupies places place-k+1 .. place
            total = sum(payouts[p - 1] for p in range(place - k + 1, place + 1))
            avg_place = (place - k + 1 + place) / 2
            for i in group:
                points[i] = total / k
                places[i] = avg_place
            place -= k
        survivors = self.alive()
        if len(survivors) == 1:
            points[survivors[0]] = payouts[0]
            places[survivors[0]] = 1
        elif survivors:
            rem = payouts[:len(survivors)]
            ev = icm([self.stacks[i] for i in survivors], rem)
            for idx, i in enumerate(survivors):
                points[i] = ev[idx]
            # expected place: use ICM over placements? keep simple: rank by chips
            ranked = sorted(survivors, key=lambda i: self.stacks[i], reverse=True)
            for r, i in enumerate(ranked):
                places[i] = r + 1
        return TournamentResult(
            points=points, final_stacks=list(self.stacks), hands=self.hand_number,
            placements=places, decisions=self.decisions, timeouts=self.timeouts,
            errors=self.errors, illegal=self.illegal, max_latency_ns=self.max_latency,
            total_latency_ns=self.total_latency, hand_records=self.records,
        )


def run_tournament(bots: Sequence[Bot], config: Config | None = None, seed: int = 0) -> TournamentResult:
    config = config or Config()
    return Table(bots, config, seed).run()
