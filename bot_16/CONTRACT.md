# Local interface contract (not an official contest interface)

`from agent import Agent`; `Agent(seed=16).act(observation) -> dict`.
One instance per table. `reset(seed=None)` resets all table-local state.
Cards are integers 0..51, `rank = card // 4 + 2`, `suit = card % 4`.
Observation is a dictionary with:
- `hero`: seat index; `button`: dealer seat index; `hand_id`: integer
- `street`: 0 preflop, 1 flop, 2 turn, 3 river
- `hole`: two card integers; `board`: 0/3/4/5 card integers
- `players`: seat-ordered dictionaries: `stack` chips BEHIND, `street_bet` chips committed on current street, `invested` committed this hand, `folded` boolean, `all_in` boolean. Eliminated seats have stack/invested zero and folded true.
- `pot`: all chips committed this hand, INCLUDING current street and folded contributions; no double counting
- `to_call`: additional chips required to match (UNCAPPED by hero stack); caller pays min(to_call, stack)
- `big_blind`: positive integer; `small_blind`: integer
- `min_raise_to`: minimum full-raise TOTAL current-street contribution
- `max_raise_to`: hero street_bet + hero stack
- `can_raise`: bool (false when no opponent can respond or action has not reopened)
- `history`: chronological actions in this hand; each has `seat`, `street`, `action` (fold/check/call/raise), `paid` (chips added), `to` (street total after action), `pot_before`. Forced blinds need not appear.
Optional: `payouts` finishing-place points (not invented by bot), `hands_remaining`, `blind_schedule`.

Action is `{'action': 'fold'}`, `{'action': 'check'}`, `{'action': 'call'}`, or `{'action': 'raise', 'amount': total_street_contribution}`.
A raise may equal max_raise_to below min_raise_to only as a short all-in, and only if can_raise. Calling an all-in is a call, never a raise. Stack chips, blind amounts and bet totals use integer units. Engine handles unmatched bet refunds, side pots, ties, reopening, odd chips, heads-up dealer blind/action order and terminal states.

Timing target is a warm `act` call including feature extraction/action normalization, excluding process import/model initialization. Also measure cold first action and report all observed >1ms decisions. Official timing/adapter semantics remain unknown.
