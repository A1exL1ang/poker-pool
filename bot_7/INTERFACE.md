# Local observation and action contract

There was no official starter kit or interface in this workspace. These names and semantics are local assumptions. `agent.Agent` re-exports the implementation and observation types; an official adapter must translate the real interface into this contract and translate the returned action back.

Create one `Agent(seed=7007, comparisons=48, budget_us=450)` per table. `reset()` clears all table state. A changed `table_id` also resets state. The agent never identifies teams or retains opponent statistics across tables. Anonymous seat numbers are used only within the current table.

`Agent.act(Observation) -> Action` is the timed entry point. `Agent.act_dict(dict) -> dict` performs normalization/validation as well. `python agent.py` accepts one JSON observation per line and emits one JSON action per line. The JSON runner is a development protocol, not a proposed official transport. Invalid JSON/contract inputs yield an `error` object. The typed entry point assumes a valid observation and uses a legal passive fallback for caught policy errors; its `errors` counter exposes those failures.

| Field | Meaning |
|---|---|
| `hole` | Exactly two hero cards. Integers `0..51`, or strings such as `As`, `Td` in dictionary input. |
| `board` | Exactly 0, 3, 4, or 5 public cards. |
| `players` | All seats relevant to the hand, including folded contributors; at most eight. |
| `hero`, `button` | Seat integers `0..7`; clockwise order is increasing seat modulo eight. Both must appear in `players`. The button may have folded. |
| `pot` | Sum of every player's whole-hand `committed`, including current-street wagers. |
| `to_call` | Actual incremental call cost, **capped by hero's stack behind**. |
| `min_raise_to` | Minimum full raise as a **total current-street contribution**, not an increment. |
| `max_raise_to` | Maximum legal total current-street contribution, at most `hero.bet + hero.stack`. |
| `legal_actions` | Subset of `fold`, `check`, `call`, `raise`, determined by the engine. The engine must enforce reopening rights. |
| `big_blind` | Current nominal big blind; positive integer. Default 20 is only a convenience. |
| `hand_id` | Changes each hand; default 0. |
| `table_id` | Changes each table; default `local`. |
| `events` | Optional public action history described below. |
| `payouts` | Optional actual descending, nonnegative finishing points for the table. Default empty means chip EV. Never assumed to be `[50,30,20]`. |
| `eligible_pot` | Optional current pot eligible for hero after calling, **excluding the new call itself**. If omitted, computed from capped whole-hand contributions. |

Every player is `Player(seat, stack, bet=0, committed=0, folded=False, all_in=False)`. `stack` means chips behind; `bet` means chips already contributed on the current street; `committed` means chips already contributed over the entire hand. All are nonnegative integers and `committed >= bet`. Busted seats may remain listed with zero stack/contribution and `folded=True`. An all-in player remains an opponent until the hand is settled. Do not send other players' hole cards.

Card encoding is rank-major, suits `c,d,h,s`: `0=2c`, `1=2d`, `2=2h`, `3=2s`, ..., `48=Ac`, `51=As`. The JSON boundary also accepts `10s` as `Ts`. Visible cards must be unique. Direct `Observation` construction takes integer cards; string normalization is only in `from_dict`.

Actions have exactly `action` and `amount`. `amount` is zero for folds, checks, and calls. A call instructs the engine to contribute the already-capped `to_call`. A raise's amount is the desired **raise-to total on this street**. Example: hero has already bet 20, faces a total of 60, and returns `Action("raise", 180)`: the engine contributes 160 more chips. An opening bet is represented as `raise` from a zero current street bet. There is no separate `all_in` action: all-in raising is `raise(max_raise_to)`; calling all-in is `call`.

When `min_raise_to > max_raise_to` and `raise` is legal, the only permitted raise is the short all-in at `max_raise_to`. If a prior short raise has not reopened action, omit `raise` from `legal_actions`. If all opponents are all-in and only calling/folding remains, omit `raise`. The agent always clamps proposals to the supplied legal bounds; it does not reconstruct official reopening rules.

`Event(id, seat, street, action, amount=0, to_call=0)` records a public action. IDs must increase monotonically **across the whole table**, not restart each hand. `street` is `preflop`, `flop`, `turn`, or `river`. `action` is one of the four action tokens. For raises, `amount` is the resulting total street wager; for calls it is the contributed amount. `to_call` is the actual amount the actor faced before acting. Do not encode blind posts as voluntary actions. Send all actions since the bot's last observation, or cumulative current-hand history; IDs prevent double counting. Only the newest 32 events are processed for predictable latency. Omitting history is supported but weakens range inference and opponent adaptation.

For side-pot odds, let `cap = hero.committed + to_call`. Eligible existing pot is `sum(min(player.committed, cap))`; the eventual call pot adds `to_call`. The bot separates further tiers by which all-in opponents are eligible. Still-active opponents are conservatively treated as able to match future wagers. Uncalled excess wagers do not count as pot odds for a short caller.

ICM utility itself is exact for the supplied stack vector. It uses an exponential-race/subset expansion, verified against independent permutation and subset-DP calculations. Applying that utility to a poker decision is approximate: for a large heads-up call, folding awards the existing pot to the opponent; winning awards it to hero; losing awards it plus hero's call to the opponent. Side-pot and multiway decisions skip this heads-up risk adjustment. Ties, future betting, and unobserved range accuracy prevent this from being a full tournament solver.

Before real integration, resolve the real card representation, stack/bet/pot conventions, action naming, total-versus-increment raise semantics, legal short-raise handling, observation lifecycle, point schedule, startup timing, and decision-time accounting. Import/load assets once before timed play if permitted. The current hardware/transport contract is unverified, so no official compatibility or strict 1 ms guarantee is claimed.
