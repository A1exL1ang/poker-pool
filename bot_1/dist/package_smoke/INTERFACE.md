# Local integration contract

The official starter kit and Agent schema were not provided. This is an explicit local interface, not a claim of official compatibility. Bind host observations and returned actions before submission.

## Decision interface (local assumptions)

```json
{
  "table_id": "example", "hand_id": 1,
  "hero": 0, "button": 0, "big_blind": 20,
  "hole_cards": ["As", "Ah"], "board": [],
  "pot": 30, "to_call": 10, "players_to_act": 1,
  "players": [
    {"seat": 0, "stack": 1990, "bet": 10, "committed": 10, "folded": false},
    {"seat": 1, "stack": 1980, "bet": 20, "committed": 20, "folded": false}
  ],
  "legal_actions": [
    {"action": "fold"},
    {"action": "call", "amount": 10},
    {"action": "raise", "min": 40, "max": 2000}
  ]
}
```

- `players` lists all players still at the table, including players folded this hand, in clockwise seating order. Seats are unique; `hero` and `button` refer to those seats. There are at most eight players. Busted players with no chips in the current pot can be omitted.
- `stack` is chips remaining behind; `bet` is total committed **this street**; `committed` is total committed **this hand**, including `bet`. `pot` includes all contributions, folded players' money, and outstanding wagers, but excludes the future call. Supply `committed` for meaningful side-pot caps and ICM. Integer chips are assumed.
- `to_call` is additional chips, capped at hero's stack internally. If absent, it is derived from street bets. `players_to_act` is how many non-all-in opponents still have an action after hero this round. If absent, seating supplies an approximation; supply it for better preflop positioning.
- Cards are rank/suit strings (`As`, `Td`, `10h`), or integers `suit*13 + rank`, with suits `cdhs` and ranks `23456789TJQKA`. This integer encoding is a local choice, not an official encoding. Boards have 0, 3, 4 or 5 cards.
- Legal actions must come from the host. `raise.min` and `raise.max` are inclusive **total street commitments**, not increments. An all-in below the normal minimum must be represented by its actual legal bounds. Exclude raises if betting has not reopened. A call's `amount` is additional chips. The bot clamps raises and never intentionally folds when it can check.
- Responses are `{"action":"check"}`, `{"action":"fold"}`, `{"action":"call","amount":10}`, or `{"action":"raise","amount":45}`. Optional legal `bet`/`all_in` labels are also accepted; the standard local engine uses `raise` for bets and all-in raises.
- Optional `eligible_pot` is the amount hero may win before adding the call. Otherwise an all-in hero's unmatchable excess is removed using contributions. Known closed side pots are evaluated by their distinct eligible opponents and normalized amounts. Unresolved future callers remain an approximation; the simulator's actual settlement is exact.
- Optional `payouts` is the **actual** descending points vector, padded with zeros, with at least as many entries as players still alive. Entries for already eliminated places do not affect decisions. With no vector, the bot uses chip EV. No official payouts have been invented.
- Optional `history` holds the last chronological events of the **current hand**. Events have table-unique `id`, `seat`, `hand_id`, `street` (`preflop`, `flop`, `turn`, `river`), normalized `action`, and optionally `facing_bet`. The timed path inspects at most 16 events. For full observations call `bot.observe(events)` chronologically outside the timed decision; repeat IDs are ignored within a bounded 512-event window. Normalize all-ins as `call` or `raise` in history.

The JSON runner also accepts `{"type":"reset","table_id":"new"}` and `{"type":"observe","events":[...]}`. Malformed card/state observations fail closed to check/fold. The host must provide a valid legal-action set and consistent monetary state.


