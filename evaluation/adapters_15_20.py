"""Public-observation adapters for immutable submissions 15 through 20.

Observation conversion is inside act() and therefore inside the host clock.
The submission loader isolates both absolute and relative local imports.
"""
from itertools import count
from common.engine import Action
from isolated_loader import load_bot_module

STREETS = ("preflop", "flop", "turn", "river")
_TABLE_IDS = count(1)


def public_events(o):
    """Reconstruct public chip bookkeeping; never consult deal/hidden cards."""
    bets = [0] * o.n_seats
    remaining = list(o.starting_stacks)
    pot = 0
    for i in range(o.n_seats):
        paid = min(remaining[i], o.ante) if o.active[i] else 0
        remaining[i] -= paid
        pot += paid
    for i, blind in ((o.sb_seat, o.sb), (o.bb_seat, o.bb)):
        paid = min(remaining[i], blind)
        remaining[i] -= paid
        bets[i] += paid
        pot += paid
    folded = [not active for active in o.active]
    current = o.bb
    street = 0
    result = []
    for e in o.history:
        if e.street != street:
            street = e.street
            bets = [0] * o.n_seats
            current = 0
        opponents = [i for i in range(o.n_seats) if i != e.seat and not folded[i]]
        target = current if any(remaining[i] > 0 for i in opponents) else max((bets[i] for i in opponents), default=0)
        result.append(dict(seat=e.seat, street=e.street, action=e.kind,
                           paid=e.paid, total=e.amount, pot_before=pot,
                           to_call_before=max(0, target - bets[e.seat]),
                           stack_before=remaining[e.seat]))
        if e.kind == "fold":
            folded[e.seat] = True
        if e.kind in ("raise", "call"):
            bets[e.seat] = e.amount
            remaining[e.seat] -= e.paid
            pot += e.paid
            current = max(current, e.amount)
    return result


def observation_for(number, o, table_id):
    """Translate one common Observation, retaining original anonymous seats."""
    if number not in range(15, 21):
        raise ValueError(number)
    h = public_events(o)
    folded = [bool(o.folded[i] or not o.active[i]) for i in range(o.n_seats)]
    call = min(o.to_call, o.stacks[o.seat])
    legal = ["fold", "call"] if call else ["check"]
    if o.can_raise:
        legal.append("raise")
    d = dict(hero=o.seat, button=o.button, hand_id=o.hand_number,
             table_id=table_id, street=o.street, hole=list(o.hole), board=list(o.board),
             pot=o.pot, big_blind=o.bb, small_blind=o.sb,
             to_call=o.to_call, min_raise_to=o.min_raise_to,
             max_raise_to=o.max_raise_to, can_raise=o.can_raise,
             payouts=list(o.payouts), hands_remaining=o.hands_left)
    players = [dict(seat=i, stack=o.stacks[i], bet=o.street_bets[i],
                    committed=o.contributions[i], folded=folded[i], all_in=o.all_in[i])
               for i in range(o.n_seats)]
    if number == 15:
        # Original preflop action order begins left of BB; heads-up button is first.
        first = o.button if sum(o.active) == 2 else (o.bb_seat + 1) % o.n_seats
        order = [(first + j) % o.n_seats for j in range(o.n_seats)]
        behind = order[order.index(o.seat) + 1:]
        d.update(stacks=list(o.stacks), in_hand=[not x for x in folded],
                 street_bets=list(o.street_bets), contributions=list(o.contributions),
                 position=sum(not folded[i] for i in behind),
                 history=[dict(seat=e['seat'], street=e['street'], action=e['action'],
                               amount=e['total'] if e['action']=='raise' else e['paid'],
                               pot_before=e['pot_before'], to_call=e['to_call_before']) for e in h])
    elif number == 16:
        d.update(players=[dict(seat=p['seat'], stack=p['stack'], street_bet=p['bet'],
                               invested=p['committed'], folded=p['folded'], all_in=p['all_in']) for p in players],
                 history=[dict(seat=e['seat'], street=e['street'], action=e['action'],
                               paid=e['paid'], to=e['total'], pot_before=e['pot_before']) for e in h])
    elif number == 17:
        d.update(players=players, to_call=call, legal_actions=legal,
                 history=[dict(seat=e['seat'], street=e['street'], action=e['action'],
                               amount=e['total'] if e['action'] in ('call','raise') else 0, paid=e['paid']) for e in h])
    elif number == 18:
        d.update(street=STREETS[o.street], to_call=call,
                 players=[dict(seat=p['seat'], stack=p['stack'], bet=p['bet'],
                               invested=p['committed'], folded=p['folded'], all_in=p['all_in']) for p in players],
                 history=[dict(seat=e['seat'], street=STREETS[e['street']], kind=e['action'],
                               amount=e['paid'], raise_to=e['total']) for e in h])
    elif number == 19:
        d.update(street=STREETS[o.street], stacks=list(o.stacks), bets=list(o.street_bets),
                 contributions=list(o.contributions), folded=folded, legal_actions=legal,
                 history=[dict(seat=e['seat'], street=STREETS[e['street']], action=e['action'],
                               amount=e['paid'], total=e['total'], pot_before=e['pot_before'],
                               to_call_before=e['to_call_before'], stack_before=e['stack_before']) for e in h])
    else:
        d.pop('hole')
        d.update(hole_cards=list(o.hole), stacks=list(o.stacks),
                 hand_start_stacks=list(o.starting_stacks), committed=list(o.contributions),
                 street_committed=list(o.street_bets), folded=folded, all_in=list(o.all_in),
                 sb_seat=o.sb_seat, bb_seat=o.bb_seat,
                 active_players=sum(not x for x in folded),
                 history=[dict(seat=e['seat'], street=e['street'], action=e['action'],
                               amount=e['paid'], to_call=e['to_call_before'],
                               pot_before=e['pot_before']) for e in h])
    return d


class Adapter:
    def __init__(self, number):
        if number not in range(15, 21):
            raise ValueError(number)
        self.number = number
        self.module = load_bot_module(number)
        self.calls = self.internal_errors = self.safety_fallbacks = self.budget_fallbacks = 0

    def reset(self, seed, seat, n_seats):
        self.bot = self.module.Agent(seed=seed)
        self.table_id = 'evaluation-table-' + str(next(_TABLE_IDS))
        self.calls = self.internal_errors = self.safety_fallbacks = self.budget_fallbacks = 0

    def act(self, o):
        d = observation_for(self.number, o, self.table_id)
        self.calls += 1
        a = self.bot.act(d)
        # errors and fallbacks are aliases for bot16; do not double count.
        self.internal_errors = max(int(getattr(self.bot, 'errors', 0)),
                                   int(getattr(self.bot, 'fallbacks', 0)))
        self.safety_fallbacks = self.internal_errors
        self.budget_fallbacks = int(getattr(self.bot, 'budget_fallbacks', 0))
        if not isinstance(a, dict):
            a = a.as_dict()
        return Action(a['action'], int(a.get('amount', 0)))

    def diagnostic_counters(self):
        return dict(internal_errors=self.internal_errors,
                    safety_fallbacks=self.safety_fallbacks,
                    budget_fallbacks=self.budget_fallbacks)
