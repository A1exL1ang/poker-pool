"""Contract-only adapters for frozen independent bots 9 through 14.

All cards retain the common rank-major encoding. Conversion and counter reads
are included in timing. Error fallbacks and intended computation-budget/native
fallbacks are reported separately; no strategy or selected parameters change.
"""
from common.engine import Action
from isolated_loader import load_bot_module

STREETS = ('preflop', 'flop', 'turn', 'river')


def public_history(o, blinds=False, numeric=False, total_calls=False):
    """Replay public contributions to supply each contract's action metadata."""
    bets = [0] * o.n_seats
    stacks = list(o.starting_stacks)
    folded = [not x for x in o.active]
    pot = 0
    out = []
    for seat in range(o.n_seats):
        paid = min(stacks[seat], o.ante) if o.active[seat] else 0
        stacks[seat] -= paid
        pot += paid
    for seat, blind, kind in ((o.sb_seat, o.sb, 'small_blind'),
                              (o.bb_seat, o.bb, 'big_blind')):
        paid = min(stacks[seat], blind)
        stacks[seat] -= paid
        bets[seat] += paid
        if blinds:
            out.append(dict(seat=seat, street=0 if numeric else 'preflop',
                            action=kind, amount=paid, paid=paid, to=bets[seat]))
        pot += paid
    street = 0
    current = o.bb
    for event in o.history:
        if event.street != street:
            street = event.street
            bets = [0] * o.n_seats
            current = 0
        other_chips = any(i != event.seat and not folded[i] and stacks[i] > 0
                          for i in range(o.n_seats))
        live_max = max((bets[i] for i in range(o.n_seats) if not folded[i]), default=0)
        facing = max(0, (current if other_chips else live_max) - bets[event.seat])
        resulting_bet = event.amount if event.kind in ('raise', 'call') else bets[event.seat]
        amount = resulting_bet if total_calls else event.amount if event.kind == 'raise' else event.paid
        out.append(dict(seat=event.seat, street=street if numeric else STREETS[street],
                        action=event.kind, amount=amount, paid=event.paid,
                        amount_paid=event.paid, to=resulting_bet,
                        pot_before=pot, to_call_before=min(facing, stacks[event.seat]),
                        stack_before=stacks[event.seat], current_bet_before=current,
                        is_all_in=event.paid > 0 and event.paid == stacks[event.seat]))
        if event.kind == 'fold':
            folded[event.seat] = True
        if event.kind in ('raise', 'call'):
            bets[event.seat] = event.amount
            stacks[event.seat] -= event.paid
            pot += event.paid
            current = max(current, event.amount)
    return out


def convert(number, o):
    call = min(o.to_call, o.stacks[o.seat])
    legal = ['fold', 'call'] if call else ['check']
    if o.can_raise:
        legal.append('raise')
    current = o.street_bets[o.seat] + o.to_call
    players = [dict(seat=i, stack=o.stacks[i], bet=o.street_bets[i],
                    committed=o.contributions[i], folded=o.folded[i], all_in=o.all_in[i])
               for i in range(o.n_seats)]
    d = dict(hero=o.seat, button=o.button, hand_id=o.hand_number,
             hole=tuple(o.hole), hole_cards=tuple(o.hole), board=tuple(o.board),
             small_blind=o.sb, big_blind=o.bb, pot=o.pot, to_call=call,
             current_bet=current, min_raise_to=o.min_raise_to,
             max_raise_to=o.max_raise_to, can_raise=o.can_raise,
             legal_actions=legal, players=players, payouts=tuple(o.payouts),
             street=STREETS[o.street], small_blind_seat=o.sb_seat,
             big_blind_seat=o.bb_seat)
    if number == 9:
        d.update(street=o.street, hand_no=o.hand_number, bb=o.bb,
                 stacks=list(o.stacks), committed=list(o.street_bets),
                 contributions=list(o.contributions), folded=list(o.folded),
                 all_in=list(o.all_in), history=public_history(o, numeric=True))
    elif number == 10:
        d.update(street=o.street, payouts=tuple(o.payouts[:sum(o.active)]),
                 history=public_history(o, blinds=True, numeric=True))
    elif number == 11:
        d.update(seat=o.seat, hand_number=o.hand_number,
                 history=public_history(o, blinds=True))
        d['players'] = [dict(p, street_bet=p['bet']) for p in players]
    elif number == 12:
        d.update(hand_number=o.hand_number, stacks=list(o.stacks),
                 contributions=list(o.contributions), street_contributions=list(o.street_bets),
                 in_hand=[o.active[i] and not o.folded[i] for i in range(o.n_seats)],
                 all_in=list(o.all_in), history=public_history(o, blinds=True, total_calls=True))
    elif number == 13:
        d.update(seat_count=o.n_seats, history=public_history(o, total_calls=True))
    elif number == 14:
        d.update(to_call=o.to_call, hand_start_stacks=tuple(o.starting_stacks),
                 history=public_history(o, blinds=True))
        d['players'] = [dict(seat=i, stack=o.stacks[i], committed=o.street_bets[i],
                             invested=o.contributions[i], folded=o.folded[i],
                             all_in=o.all_in[i], starting_stack=o.starting_stacks[i])
                        for i in range(o.n_seats)]
        for e in d['history']:
            if e['action'] == 'small_blind':
                e['action'] = 'post_sb'
            elif e['action'] == 'big_blind':
                e['action'] = 'post_bb'
    else:
        raise ValueError(number)
    return d


class Adapter:
    def __init__(self, number):
        if number not in range(9, 15):
            raise ValueError(number)
        self.number = number
        self.module = load_bot_module(number, 'bot' if number == 9 else 'agent')
        self.bot = None
        self.calls = self.internal_errors = self.budget_fallbacks = self.native_fallbacks = 0

    def reset(self, seed, seat, n_seats):
        self.bot = self.module.Agent(seed=seed)
        self.calls = self.internal_errors = self.budget_fallbacks = self.native_fallbacks = 0

    def _counters(self):
        n, b = self.number, self.bot
        if n == 9:
            return b.errors, 0, 0
        if n == 10:
            return b.failures, 0, 0
        if n == 11:
            return b.diagnostics['errors'], 0, 0
        if n == 12:
            return b.fallbacks, 0, 0
        if n == 13:
            return b.fallbacks, 0, b.native_errors
        return 0, b.diagnostics['budget_fallbacks'], b.diagnostics['equity_native_fallbacks']

    def act(self, o):
        self.calls += 1
        d = convert(self.number, o)
        before = self._counters()
        try:
            result = self.bot.decide(d) if self.number == 14 else self.bot.act(d)
        finally:
            after = self._counters()
            self.internal_errors += max(0, after[0] - before[0])
            self.budget_fallbacks += max(0, after[1] - before[1])
            self.native_fallbacks += max(0, after[2] - before[2])
        return Action(result['action'], int(result.get('amount', 0)))
