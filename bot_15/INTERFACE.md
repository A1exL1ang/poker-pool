# Local interface v1 (not an official contest interface)
Cards: integer 0..51, rank=card//4 (0=2,12=A), suit=card%4.
Observation is a dict:
hero: seat index; button: seat index; hand_id: int; street: 0 preflop,1 flop,2 turn,3 river.
hole: 2 ints; board: 0/3/4/5 ints; stacks: list of remaining integer stacks.
in_hand: list of bool (not folded, including all-ins); street_bets: chips committed on current street.
contributions: total chips committed to hand. pot: sum contributions BEFORE action, including live bets.
big_blind, small_blind: integers. to_call: maximum street_bets minus hero street_bet, NOT capped to stack. Exception: a short all-in big blind still imposes the nominal big-blind bring-in preflop, so the amount to call can exceed the largest actually posted street bet.
min_raise_to: minimum full legal raise total on street. max_raise_to: hero street_bet+hero stack.
can_raise: bool, false when action not reopened by short all-in or no opponent can respond.
position: number of active seats acting after hero preflop. Supply this explicitly with dead blinds; the optional fallback assumes normal live-blind ordering.
history: current-hand cumulative, chronologically ordered list of voluntary action dicts with seat, street, action ('fold','check','call','raise'), amount (raise TO or call chips), optional pot_before and optional to_call (uncapped amount faced BEFORE that action). Exclude forced blind/ante events. Start a fresh list at each hand_id.
payouts: optional actual finishing-position points for all original seats, descending and padded with zeros for unpaid places; absent means chip EV. No official payout assumptions.
Action: dict {"action": "fold"|"check"|"call"|"raise", "amount": int}. Fold/check amount 0; call amount min(to_call,stack); raise amount TOTAL street commitment. A raise may be below min_raise_to only if all-in at max_raise_to.
Agent in bot/agent.py exposes Agent(seed=...).act(observation)->Action and reset(). Constructor/loading outside decision timing. No network, stdlib runtime. Adapter integration required once official API exists.

Call reset() for every new table; hand_id must distinguish hands within the table. Retain eliminated seats as zero-stack entries or explicitly remap all seat fields consistently. Do not expose other players' private cards, future community cards, or team identities. Run bot.protocol.validate_observation during adapter integration outside the timed decision path.
