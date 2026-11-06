# Local contract (not an official contest interface)

`from agent import Agent`; instantiate once per table with `Agent(seed=...)`.
Call `act(observation)` for a dictionary action. `reset(seed=...)` clears all table-local
learning. Cards are integers `4*(rank-2)+suit`, rank 2..14, suit 0..3.
A new Agent is required for a new anonymous table; there is no cross-table identity.

Observation keys:
- `hand_id`: table-local hand number; `street`: preflop/flop/turn/river.
- `hero`, `button`: seat indices; `hole` two hero cards; `board` 0/3/4/5 cards.
- `stacks`: chips remaining, excluding commitments.
- `bets`: each player's current-street commitment.
- `contributions`: total commitments in this hand, including current-street bets.
- `folded`: folded or eliminated seats; `pot`: sum of all hand commitments.
- `to_call`: highest live street bet minus hero commitment, before stack clipping.
- `min_raise_to`: minimum full raise TOTAL current-street commitment.
- `max_raise_to`: hero street bet plus remaining stack.
- `can_raise`: whether betting is reopened and there is an opponent able to respond.
- `legal_actions`: allowed action strings (includes short all-in `raise` when legal).
- `small_blind`, `big_blind`: current blind sizes.
- `history`: public actions in THIS hand only, in order. Events include seat, street,
  action, amount (newly contributed), total (street contribution), pot_before,
  to_call_before and stack_before. Blind events are action `blind`.
- `payouts`: OPTIONAL explicit finishing-order points, highest first. None means
  chip-equity objective. Any values used in local benchmarks are hypothetical.

Actions are `{ "action": "fold"|"check"|"call"|"raise", "amount": integer }`.
Only a raise uses `amount`: it is a TOTAL street commitment, NOT a raise increment
and NOT the amount of new chips. A call is capped by the acting player's stack.
An adapter to any eventual official interface must translate and validate all these
semantics. No official compatibility is claimed.

Timing is in-process `Agent.act` only. Import/model loading, process creation,
JSON parsing and observation construction are measured separately where applicable;
the unavailable official clock boundary cannot be assumed.
