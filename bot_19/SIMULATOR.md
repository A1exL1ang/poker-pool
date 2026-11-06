# Independent local simulator

`sim.py` is a newly written test environment for standard no-limit Texas
Hold'em. It is not the missing official starter kit, and its observation and
action schema does not claim official compatibility.

The contest rules specify eight starting players, 2,000 chips each, initial
10/20 blinds, a level increase every 25 hands, and a 200-hand cap. They do not
supply the subsequent blind amounts or the place payouts. The default local
blind sequence is **10/20, 15/30, 25/50, 40/80, 60/120, 100/200, 150/300,
250/500**, with no ante. This sequence is an experimental assumption.
`Tournament` accepts another sequence. `payouts=None` is the default and
produces `scores=None`; an explicit first-to-last payout vector is required
to compute local point scores.

## API

```python
from sim import Tournament

# Construct independent agent objects for every seat/table.
result = Tournament(
    agents,
    seed=123,
    max_hands=200,
    enforce_time_limit=True,
).run()
print(result.stacks, result.chip_shares, result.stats.overruns)
```

`HandState` also supports stepping with `observation()` and `apply(action)`.
`apply(..., strict=True)` raises on an illegal action; normal mode records an
invalid action and checks if available, otherwise folds.

Every bot has `act(observation) -> action`. An optional no-argument `reset()`
is called at table start. Observations are detached dictionaries. They contain
the acting player's two hole cards, revealed board, public stack and commitment
arrays, blind/button positions, public action history, and legal action bounds.
They never contain the other hole cards, deck order, burn cards, or future board.
Cards are integers `0..51`, with rank `card // 4 + 2` and suit `card % 4`.

Actions are `{"action": "fold"}`, `{"action": "check"}`,
`{"action": "call"}`, or `{"action": "raise", "amount": integer}`.
For a raise, `amount` is the **total current-street commitment**, including
the actor's chips already bet on that street. Calls automatically clip to the
remaining stack. A raise below `min_raise_to` is legal only if it equals
`max_raise_to`, making the actor all-in. Raising is unavailable when betting
has not reopened or no opponent has chips available to contest another bet.

| Observation field | Meaning |
| --- | --- |
| `hand_id`, `street`, `hero`, `button` | Zero-based table hand number, street name, acting seat, button position |
| `hole`, `board` | Own cards and current public board |
| `stacks` | Chips remaining behind, after all commitments |
| `bets` | Current-street commitments |
| `contributions` | Total current-hand commitments, after refunds |
| `pot` | Sum of all contributions, **including current-street bets** |
| `to_call` | Uncapped amount required to call, before stack clipping |
| `min_raise_to`, `max_raise_to` | Minimum full raise total; maximum total using actor's entire stack |
| `can_raise`, `legal_actions` | Actual available decisions, including reopening restrictions |
| `active` | Seats that started this hand with chips |
| `folded` | Folded and nonparticipating seats |
| `hand_start_stacks` | Public stacks before blind posts |
| `small_blind`, `big_blind` | Nominal blind amounts |
| `small_blind_seat`, `big_blind_seat` | Blind positions; SB position can be vacant under the dead-button convention |
| `history` | Public events for this hand only; no terminal private data |
| `street_raise_count`, `last_aggressor` | Aggression on this street only |
| `actor_order` | Current pending actors in clockwise order, beginning with hero |
| `payouts` | Included only when explicitly supplied to the simulator |

History decision events have `seat`, `street`, `action`, `amount`, `total`,
`pot_before`, `to_call_before`, and `stack_before`. Here `amount` is the number
of **additional** chips committed; `total` is the resulting street commitment.
This differs deliberately from the raise action's `amount` field. Blind posts
use `action="blind"` with `blind="small"` or `"big"`. Uncalled-chip returns
use `action="refund"`. Neither is a voluntary decision.

## Mechanics and conventions

The betting and odd-chip implementation follows the public
[Poker TDA rules](https://www.pokertda.com/view-poker-tda-rules/), especially
minimum raises, cumulative incomplete all-ins, heads-up order, and odd chips.
These are independent simulator choices, not organizer-confirmed details.

* Heads-up, the button posts the small blind and acts first preflop; the big
  blind acts first on later streets. The big blind receives the first hole card.
* A short posted big blind does not reduce the nominal preflop bring-in while
  multiple players can still bet. With only one non-all-in player, the engine
  requires only matching the actual opposing wager and then runs out the board.
* Incomplete all-ins do not change the full raise increment. Reopening is
  checked separately for each actor; multiple short all-ins can cumulatively
  reopen one player's action while leaving another player's action closed.
* The big blind advances to the next surviving seat. In multiway play the
  prior small-blind position becomes the button and the prior big-blind
  position becomes the small blind; either of those positions can be vacant.
  On transition to heads-up, the button is adjusted so nobody posts the big
  blind consecutively. Vacant seats receive no cards and take no action.
* Uncalled money is returned at the end of a betting round. Side pots are
  built from contribution levels, include folded chips, and restrict winners
  by eligibility. Each pot is split separately; odd chips go clockwise from
  the button among tied winners.
* Simultaneous eliminations rank larger hand-start stacks higher. Equal
  hand-start stacks share the affected places and local awards equally.
* At the hand cap, exact ICM allocates the remaining explicitly supplied
  place awards. Eliminated seats retain their previously assigned awards.

## Evidence and limitations

Run `python -m unittest discover -s tests -p test_sim.py -v`. The tests include
adversarial minimum-raise/reopening fixtures, heads-up and dead-button states,
multiple side pots, folded dead money, uncalled refunds, split-board odd chips,
privacy/snapshot checks, table resets, timeout/error fallback, deterministic
replays, simultaneous elimination ties, 350 randomized legal hands, and 80
randomized eight-seat tournaments. ICM is checked against an independent
permutation enumeration and analytic two-player cases.

Timing uses `time.perf_counter_ns()` around `agent.act` only; observation
construction, validation, ICM, and simulation mechanics are outside that
interval. All measured overruns are retained in `result.stats`, even when
enforcement is disabled. With enforcement enabled an overrun becomes a check
or fold **after the function returns**. This local Python process does not
preempt a stuck agent and is not a security sandbox. Operating-system stalls
can affect wall-clock measurements. Official hardware, timer semantics,
interface details, payouts, and later blind amounts remain unknown.
