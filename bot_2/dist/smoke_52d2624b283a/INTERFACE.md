# Local integration contract

This is a documented assumption, not the competition's unpublished Agent API. The implementation lives in `pokerbot/model.py`. Translate the official observation and action objects at this boundary when available.

```python
from pokerbot import Agent, Observation
agent = Agent()  # initialize outside the timed decision
observation = Observation.from_dict(observation_dict)
action = agent.act(observation)
result = action.as_dict()
```

The typed hot path trusts a valid observation. JSON conversion validates input and rejects unknown fields. Invalid JSON/state is diagnosed on stderr and gets a conservative check/fold response. Fix integration errors instead of relying on fallback.

## Cards, seats and chips

- Cards are integers `rank * 4 + suit`: ranks `23456789TJQKA`, suits `cdhs`. The JSON adapter also accepts `As`, `Td`, `10h`. Supply exactly two private cards and 0/3/4/5 board cards. Never supply opponent private cards.
- Numeric seat IDs increase clockwise. Include every original table seat, including folded and eliminated players. `hero` is the acting seat; `button` is a live seat. Heads-up, the button posts the small blind and acts first preflop.
- `Player.stack` is remaining chips, `street_bet` is this street's contribution, and `invested` is the cumulative contribution this hand, including `street_bet`. Keep folded contributions represented. `sitting_out` denotes an inactive seat.
- `pot` counts every contributed chip exactly once, including current street bets. The strategy caps inaccessible side-pot contributions when pricing calls. Do not supply a pot that already counts current bets twice.

## Legal actions

- `to_call` is additional call cost capped at the hero's remaining stack. With only all-in opponents, do not require matching inaccessible nominal blind/raise excess.
- `max_raise_to` equals `hero.street_bet + hero.stack`. `min_raise_to` is the host's full minimum target. A permitted short all-in may have `max_raise_to < min_raise_to`.
- `can_raise` and `legal` must reflect actual reopening status and host permissions. Legal action names are `fold`, `check`, `call`, `raise`. `raise` also means an opening bet. Translate any official bet/raise/all-in distinctions in the adapter.
- A raise action's `amount` is the **total street contribution**, not a raise increment or amount paid now. Other actions use `amount: 0`; the host calculates call payment. An all-in call returns `call`; an all-in raise returns `raise` at the maximum target.
- The action gate handles checking instead of folding when free, full minimum raises, short all-ins, closed raising, and raise attempts that must become calls. It does not reconstruct official legal permissions from incomplete state.

## History and tournament context

- `history` is the full current hand in chronological order. Streets are 0/1/2/3. For raises and calls, `Event.amount` is that seat's total street contribution after acting. Include voluntary actions and omit blind posts. History may be omitted, with reduced range inference and adaptation.
- `hand_id` changes each hand. Anonymous seats are stable only within a table. Set a new `table_id` or call `Agent.reset()` before each new table. State never identifies other teams or persists across tables.
- Supply the actual nonnegative, nonincreasing table-point `payouts`, including zeros for unpaid places. The vector covers all original seats. This activates ICM risk adjustment; omitting it selects chip EV. Do not substitute experimental test payouts for official ones.
- `hands_remaining` is accepted but unused by the current policy. Blind amounts come from each observation; the bot does not impose an invented blind schedule.
- Inputs contain no clock-dependent strategic randomness. A private Agent seed and state-derived mixing support reproducible play. Preload imports and assets before the timing window; startup reads/checks files, decisions do not.
