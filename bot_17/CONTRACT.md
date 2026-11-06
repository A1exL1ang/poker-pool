# Local integration contract (not an official interface)

`from agent import Agent`; `Agent(seed=17)`; `act(observation: dict) -> dict`.
All chips are nonnegative integers. Seats remain stable within a table. Cards are integers 0..51: `4 * rank + suit`, rank 0=2 through 12=Ace, suit 0=clubs, 1=diamonds, 2=hearts, 3=spades. A player never observes hidden opponent cards or future cards.

Required observation fields:
- `hand_id`: integer; `hero`: seat; `button`: seat; `street`: 0=preflop, 1=flop, 2=turn, 3=river.
- `hole`: two cards; `board`: 0/3/4/5 public cards.
- `players`: list of dictionaries with `seat`, `stack` (uncommitted remaining chips), `bet` (this street), `committed` (whole hand including bet), `folded`, `all_in`.
- `pot`: all committed chips including current bets. `big_blind`, `small_blind`.
- `to_call`: incremental chips hero must call, capped at hero stack.
- `min_raise_to`: minimum full raise target in total street chips; `max_raise_to`: hero bet + stack.
- `can_raise`: includes reopening restrictions; short all-in raises may be below min_raise_to.
- `legal_actions`: list containing applicable names among `fold`, `check`, `call`, `raise`.
- `history`: current hand actions, each `{seat, street, action, amount, paid}`. `amount` is resulting street bet for call/raise, zero for fold/check; `paid` is incremental chips paid. Blinds optionally appear with action `blind`.

Optional fields: `payouts` (actual nonincreasing points for finishing places, never assumed official); `hands_remaining`; `table_id`. Without payouts strategy uses chip utility with conservative survival adjustment; it must not fabricate a payout vector.

Returned action is `{"action": "fold"|"check"|"call"}` or `{"action":"raise", "amount": N}`. A raise amount is TOTAL street contribution, NOT an increment. Raise to `max_raise_to` is all-in. If shorter than `min_raise_to`, it is legal only when exactly all-in and raising is permitted. Bet is represented as raise from zero. Call is capped by stack. `Agent.reset(seed=None)` resets table state. State never identifies teams; only anonymous seats within the current table.

Timing target: warm in-process `Agent.act` wall time <1 ms including observation handling; import, model loading, and first construction are separately reported. No network or filesystem operations during act. Missing or inconsistent input returns check when legal, otherwise fold. Official integration needs an adapter and revalidation when the real starter kit is supplied.
