# bot_12

An independent, offline Python bot for standard no-limit Texas Hold'em. This is
a local experimental implementation, **not a contest submission**. No official
starter kit, Agent API, payout vector or full blind schedule was supplied.
The collection entry point is `agent.Agent.act(observation)`; an official adapter
will need to map the real API to the documented contract below.

## Run

Python 3.11 or newer; **no third-party gameplay dependencies**. Import the module
and initialize one `Agent` outside the decision timer, then call `act` in process.
Create a fresh instance, or call `reset()`, at each new anonymous table. Optional
`run_bot.py` reads one observation per JSON line and writes one action per line.
JSON transport, process startup and model loading are not included in `act` time.

```python
from agent import Agent
bot = Agent(seed=120012)
action = bot.act(observation)
```

`python -m unittest discover -s tests -v` runs the dependency-free regression
suite. Benchmark commands and measured limitations are in `measurements/README.md`.
`python package_bot.py` builds `dist/bot_12.zip` and verifies an extracted copy in
an isolated Python process. Extract the archive before importing; it is not a
directly importable zip package.
The archive contains the runtime and its API documentation. The full workspace
also contains the simulator, tests, training scripts and measurement reports.

## Local observation contract

Cards are integers `4*(rank-2)+suit`, rank 2 through 14, suits clubs=0,
diamonds=1, hearts=2, spades=3. Example: ace of spades is 51. Card order within
the hole cards and board does not affect hand evaluation. Arrays retain their
original seat indices after eliminations. All chip amounts are integers.

- `hole`: hero's two cards; `board`: zero, three, four or five public cards.
- `hero`, `button`: seat indices. `street`: `preflop`, `flop`, `turn` or `river`.
- `stacks`: remaining chips, excluding amounts already contributed.
- `contributions`: each seat's total contribution during this hand, **including
  folded money**. `street_contributions`: contribution on this betting street.
- `in_hand`, `all_in`: boolean arrays. Folded/eliminated seats have `in_hand=false`;
  an all-in player remains in the hand and may win eligible pots.
- `pot`: sum of the current hand contributions, including current bets and any
  unmatched overbet that has not yet been returned.
- `current_bet`: amount to match, normally the highest street contribution.
  A short blind can leave a nominal full-big-blind bring-in larger than every
  actual contribution.
- `to_call`: the smaller of remaining hero stack and amount required to match.
- `min_raise_to`: nominal minimum full raise **street total**, which can exceed
  hero's stack. `max_raise_to`: hero's street contribution plus remaining stack.
- `legal_actions`: list drawn from `fold`, `check`, `call`, `raise`. The engine
  must exclude raising when betting is not reopened or no opponent can contest
  another bet. If raising is legal and `max_raise_to < min_raise_to`, only that
  short all-in target is allowed. A free option should include `check`.
- `small_blind`, `big_blind`, `hand_number`: current blinds and zero-based hand
  counter within this table.
- `history`: current hand's visible actions in chronological order, across all
  streets; entries contain `seat`, `street`, `action`, `amount` and `paid`.
  `amount` for a raise is its street total. Blind posts are not voluntary raises.
  History must grow monotonically within a hand. No other hole cards or future
  community cards are observed.
- Optional `payouts`: the actual, descending finishing-point vector, first place
  first. Missing trailing places are zero. If omitted, decisions use chip odds;
  **the bot does not invent contest payouts**. Unknown blind changes are not
  forecast as official facts.

The output is `{"action":"fold"}`, `{"action":"check"}`,
`{"action":"call"}`, or `{"action":"raise","amount":N}`. `N` is the
hero's total contribution **on the current street**, not an additional amount
and not the total contribution across the hand. Call amounts are chosen by the
engine. The policy clamps its intended raises to the supplied legal bounds.
Malformed dict observations take a counted check/fold fallback where possible;
this cannot compensate for an incorrectly mapped interface.

## Strategy and offline assets

Preflop play uses an original combination-weighted hand ordering, position,
effective stacks, eight precomputed heads-up range-equity curves, small-stack shove
EV approximations, and constrained value/blocker reraises. Postflop play uses a
compact trained equity predictor on flop/turn, exact fixed-range heads-up river
equities and ties, price and effective pot size, draw ownership,
bet-size-dependent range selection, multiway corrections and controlled bluffs.
Anonymous seat statistics have Bayesian priors and reset at each table. The
policy never identifies teams and shares no state with opponents.

The evaluator computes exact best-five ranks for five, six and seven cards.
Offline labels come from an original C++ evaluator and Monte Carlo generator.
Two regression targets estimate heads-up equity against uniform legal hole
cards and against a fixed top-20% preflop range. Range classes are ordered by
independently simulated equity versus uniform hands and weighted by their card
combination counts. The preflop ranges are approximately 1%, 2.5%, 5%, 10%, 20%,
35%, 50% and 100%; whole classes make the boundaries discrete.

River equity sums legal rank classes using a small exact rank table, then
corrects for flush combinations and preflop range membership. This avoids
sampling on every river texture. The two heads-up ranges remain fixed; a
multiway split-pot adjustment uses an explicitly approximate independence model.
Within a hand, unchanged hole cards and board reuse their own equity analysis.

`assets/equity_predict.py` is generated ordinary Python; no scientific library,
native binary, compilation, network request or live sampling is required for
gameplay. The JSON forest is an auditable fallback representation. Training uses
NumPy/scikit-learn only offline; see `training/README.md` and the reports there.
The current predictor has 140 joint-output trees. Its data split is 300,000
training states, a training-only complexity
validation split, and 15,000 separately seeded final held-out states. Postflop
training labels use 1,024 samples per target; final held-out labels use 4,096.

When actual payouts are present, a small-table exact ICM routine adjusts large
heads-up call thresholds. It explicitly excludes multiway/other-all-in contexts
that do not fit the binary outcome approximation. The ICM calculation itself is
exact for its supplied stack outcomes; assigning all of the contested pot to the
lone opponent on a fold is a modeling assumption. Guaranteed shared-nuts boards
bypass a binary win/loss threshold and call for their guaranteed chop.

## Local simulator assumptions

The supplied contest rules specify eight players, 2,000 starting chips, initial
10/20 blinds, changes every 25 hands, and a 200-hand cap with remaining points
distributed using ICM. The local simulator implements those mechanics with
configurable **synthetic** blind increases and payout vectors. Its default
doubling schedule and `(0.5,0.3,0.2,0,0,0,0,0)` points are experimental scenarios,
not official contest values. It also tests gentler/faster schedules and alternate
point vectors. No rake, antes or rebuys are assumed. The moving live button is a
local convention; the transition to heads-up advances the big blind to avoid
charging the same surviving player twice consecutively. Simultaneous bustouts
are ranked by pre-hand stack, splitting tied ranks.

Tests cover best-five evaluation, side pots and refunds, ties and odd chips,
heads-up betting order, nominal bring-ins with short blinds, minimum raises,
short and cumulative all-in reopening, legal action semantics, chip and point
conservation, ICM, privacy of observations, guaranteed chops, and table resets.

## Limits

This is a measured heuristic policy, not a solved NLHE strategy. Heads-up equity
models do not fully describe correlated multiway outcomes or a posterior range
after a particular betting line. The multiway and bet-selection adjustments,
equity realization, shove approximation and positional ordering remain
approximations. Opponent adaptation is deliberately conservative and uses only
public actions that this bot has observed. Scripted sparring policies and frozen
self-play checkpoints are limited proxies for the unknown contest field.

The 1 ms rule is a wall-clock requirement. Measured percentiles and actual
overrun counts are reported; Python and a shared OS cannot guarantee a hard
deadline under scheduling pauses. The simulator can apply the contest's
automatic check/fold behavior on overruns. Official hardware, initialization
timing and adapter overhead remain unverified. Further work should start with
the actual starter kit, payouts, blind increments and sandbox once available.

All code and generated models were developed independently in this workspace.
No other bot or private poker repository was inspected. Only new, private,
bounded compute jobs were used; no pre-existing allocation or shared SSH master
was consumed or changed. Nothing was pushed, published or submitted.

Public research/mechanics references, not copied implementations:
[Brown and Sandholm, multiplayer poker](https://doi.org/10.1126/science.aay2400),
[ICM and risk aversion](https://arxiv.org/abs/0911.3100), and
[Poker TDA rules](https://www.pokertda.com/view-poker-tda-rules/).
