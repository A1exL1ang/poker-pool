"""Tournament policy and a deliberately explicit, unofficial observation API.

No network, subprocesses, model inference, or disk I/O occurs inside act().
See README.md for amounts, legal actions and official-interface limitations.
"""
import math
from .evaluator import parse_card, packed, evaluate_bits
from .equity import (EquityEstimator, OPEN_PERCENTILE, STRENGTH,
                     hand_class, preflop_equity, draw_quality, board_texture)
from .icm import call_threshold, hero_icm


class Opponent:
    __slots__ = ("hands", "voluntary", "raises", "bets", "folds", "responses", "last_hand", "vp_hand", "raise_hand")
    def __init__(self):
        self.hands = 12
        self.voluntary = 3.
        self.raises = 2.
        self.bets = 2.
        self.folds = 2.
        self.responses = 4.
        self.last_hand = self.vp_hand = self.raise_hand = None

    @property
    def vpip(self):
        return self.voluntary / self.hands

    @property
    def fold_rate(self):
        return self.folds / self.responses


class Agent:
    def __init__(self, seed=0xB071):
        self.initial_seed = seed
        self.equity = EquityEstimator(seed)
        self.table_id = None
        self.models = {}
        self.seen_events = set()
        self.event_queue = []
        self.hand_id = None
        self.roles = {}
        self.preflop_raises = 0
        self.last_raiser = None
        self.last_info = {}

    def reset(self, table_id=None):
        self.table_id = table_id
        self.models.clear()
        self.seen_events.clear()
        self.event_queue.clear()
        self.hand_id = None
        self.roles.clear()
        self.preflop_raises = 0
        self.last_raiser = None
        self.equity.seed = self.initial_seed & 0xffffffff or 1
        self.last_info = {}

    def observe(self, events):
        """Optional bounded event feed. IDs must be unique within this table."""
        for event in events:
            if "id" not in event:
                continue
            key = event["id"]
            if key in self.seen_events:
                continue
            hand = event.get("hand_id")
            if hand is not None and hand != self.hand_id:
                self._start_hand(hand)
            self.seen_events.add(key)
            self.event_queue.append(key)
            if len(self.event_queue) > 512:
                self.seen_events.remove(self.event_queue.pop(0))
            seat = event.get("seat")
            if seat is None:
                continue
            m = self.models.setdefault(seat, Opponent())
            action = event.get("action", "")
            street = event.get("street", "")
            if street == "preflop":
                if m.last_hand != hand:
                    m.hands += 1
                    m.last_hand = hand
                if action in ("call", "raise", "bet", "all_in") and m.vp_hand != hand:
                    m.voluntary += 1
                    m.vp_hand = hand
                if action in ("raise", "bet", "all_in"):
                    if m.raise_hand != hand:
                        m.raises += 1
                        m.raise_hand = hand
                    if hand == self.hand_id:
                        self.preflop_raises += 1
                        self.last_raiser = seat
                        self.roles[seat] = "raise" if self.preflop_raises == 1 else "reraise"
                elif action == "call" and hand == self.hand_id:
                    self.roles.setdefault(seat, "call")
            if event.get("facing_bet", False):
                m.responses += 1
                m.folds += action == "fold"
            if street != "preflop" and action in ("bet", "raise", "all_in"):
                m.bets += 1

    def act(self, observation):
        """Return {action, amount?}; amount is total street commitment for raises."""
        legal = observation.get("legal_actions", []) if isinstance(observation, dict) else []
        specs = {}
        try:
            for item in legal:
                if isinstance(item, str):
                    specs[item] = {}
                else:
                    specs[item["action"]] = item
            return self._act(observation, specs)
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError, AttributeError, StopIteration):
            self.last_info = {"fallback": "invalid observation"}
            return {"action": "check" if "check" in specs else "fold"}

    def _act(self, obs, legal):
        table = obs.get("table_id")
        if table != self.table_id:
            self.reset(table)
        hand = obs.get("hand_id", 0)
        if hand != self.hand_id:
            self._start_hand(hand)
        # Hosts can call observe outside timed decisions. This convenience path
        # is explicitly bounded; feed complete events through observe instead.
        self.observe(obs.get("history", [])[-16:])
        hole = [parse_card(c) for c in obs["hole_cards"]]
        board = [parse_card(c) for c in obs.get("board", [])]
        if len(hole) != 2 or len(board) not in (0, 3, 4, 5) or len(set(hole + board)) != len(hole + board):
            raise ValueError("invalid cards")
        players = obs["players"]
        seat = obs["hero"]
        me = next(p for p in players if p["seat"] == seat)
        stack = int(me["stack"])
        bet = int(me.get("bet", 0))
        bb = max(1, int(obs["big_blind"]))
        pot = int(obs["pot"])
        if min(stack, bet, pot) < 0:
            raise ValueError("negative chips")
        live = [p for p in players if not p.get("folded", False) and (p["stack"] > 0 or p.get("committed", 0) > 0 or p.get("bet", 0) > 0)]
        foes = [p for p in live if p["seat"] != seat]
        if not foes:
            return self._finish("check", 0, legal, stack, bet, 0)
        current = max(int(p.get("bet", 0)) for p in live)
        cost = min(stack, max(0, int(obs.get("to_call", current-bet))))
        # If hero is short, discard uncontestable contributions above hero's
        # maximum matched investment. Explicit eligible_pot is preferable when
        # the host already computes side pots.
        eligible_pot = pot
        if "eligible_pot" in obs:
            eligible_pot = max(0, int(obs["eligible_pot"]))
        elif "committed" in me and cost == stack:
            cap = int(me["committed"]) + cost
            eligible_pot -= sum(max(0, int(p.get("committed", 0))-cap) for p in players)
            eligible_pot = max(0, eligible_pot)
        order = [p["seat"] for p in players if p["stack"] > 0 or p.get("committed", 0) > 0]
        button = obs.get("button", order[-1])
        bindex = order.index(button) if button in order else len(order)-1
        hindex = order.index(seat)
        # Postflop relative position among contenders, independent of seat gaps.
        turn_order = sorted((p["seat"] for p in live), key=lambda s: (order.index(s)-bindex-1) % len(order))
        can_act = [s for s in turn_order if next(p for p in live if p["seat"]==s)["stack"] > 0]
        in_position = not can_act or can_act[-1] == seat
        if not board:
            bb_index = (bindex+1) % len(order) if len(order)==2 else (bindex+2) % len(order)
            pre_order = sorted((p["seat"] for p in live),key=lambda s:(order.index(s)-bb_index-1)%len(order))
            default_behind = sum(next(p for p in live if p["seat"]==s)["stack"]>0 for s in pre_order[pre_order.index(seat)+1:])
        else:
            default_behind = max(0,len(turn_order)-turn_order.index(seat)-1)
        behind = int(obs.get("players_to_act", default_behind))
        behind = max(0, min(len(foes), behind))
        k = hand_class(*hole)
        max_opp = max(int(p["stack"]) + int(p.get("bet", 0)) for p in foes)
        effective = min(stack + bet, max_opp) / bb
        active_n = len(order)
        threshold = cost / max(1., eligible_pot + cost)
        payouts = obs.get("payouts")
        icm_used = False
        if payouts is not None and cost > 0 and len(foes) == 1 and (cost == stack or foes[0]["stack"] == 0 or len(board) == 5) and "eligible_pot" not in obs:
            # Evaluate actual payout utility only for terminal calls with no
            # side pots. Never fabricate the unknown contest payout schedule.
            alive = [p for p in players if p["stack"] > 0 or p.get("committed", 0) > 0]
            if 1 < len(alive) <= 8 and len(payouts) >= len(alive) and all(payouts[i] >= payouts[i+1] >= 0 for i in range(len(payouts)-1)):
                stacks = [int(p["stack"]) for p in alive]
                hi = next(i for i,p in enumerate(alive) if p["seat"] == seat)
                vi = next(i for i,p in enumerate(alive) if p["seat"] == foes[0]["seat"])
                # Side pots require an explicit host adapter; skip exact ICM
                # if a called contribution cannot be matched by villain.
                if all(int(p.get("committed", 0)) <= int(me.get("committed", 0)) + cost for p in alive):
                    threshold = call_threshold(stacks, hi, vi, pot, cost, payouts)
                    icm_used = True
        common = (legal, stack, bet, cost)
        foes.sort(key=lambda p: p.get("bet", 0), reverse=True)
        layers=self._pot_layers(players,me,foes,cost,obs.get("players_to_act"))
        if not board:
            return self._preflop(obs, hole, k, foes, behind, active_n, in_position, effective, bb, eligible_pot, threshold, icm_used, common,layers)
        # First range is the current bettor and receives action conditioning.
        # Conditioning every opponent on the same bet would over-tighten a
        # multiway pot and multiply importance-weight variance needlessly.
        fractions = [self._range(p["seat"], active_n, False) for p in foes]
        pressure = cost / max(bb, eligible_pot-cost)
        aggression = min(.95, .45 + .35 * pressure) if cost else .12
        eq, error = self.equity.estimate(hole, board, fractions, aggression,layers=layers)
        # Detect the board itself being the absolute nuts: nobody can improve
        # beyond an ace-high straight flush; Monte Carlo smoothing must not
        # manufacture folds in a guaranteed split.
        if len(board) == 5 and evaluate_bits(packed(board)) == (0x8000000 | 13):
            eq = sum(w/(1+mask.bit_count()) for mask,w in layers) if layers else 1./(len(foes)+1)
            error = 0.
        draw = draw_quality(hole, board)
        texture = board_texture(board)
        category = evaluate_bits(packed(hole+board)) >> 24
        spr = min(stack, max(int(p["stack"]) for p in foes)) / max(1., eligible_pot)
        self.last_info = {"street": len(board), "equity": eq, "uncertainty": error, "threshold": threshold, "icm": icm_used, "draw": draw,"side_pot_layers":len(layers) if layers else 0}
        if cost:
            # Future-street realization and reverse implied odds; all-in and
            # river calls use raw showdown equity and actual pot odds.
            terminal = cost == stack or all(p["stack"] == 0 for p in foes) or len(board) == 5
            realization = 1. if terminal else .94 if in_position else .86
            if draw > .5 and not terminal:
                realization = min(1., realization + .05)
            value = eq * realization
            # Tiny samples should not alone trigger a huge value raise.
            raise_cutoff = .73 if len(foes) == 1 else .64
            if eq > raise_cutoff and eq-error*.45 > threshold+.16 and (category >= 2 or spr < 1.2):
                target = bet + cost + int((eligible_pot+cost) * (.72 if texture > .4 else .57))
                if spr < 1.35 or target-bet > stack*.65:
                    target = stack+bet
                return self._finish("raise", target, *common)
            # Semi-bluff raises are rare, heads-up, and backed by actual draws.
            if draw > .65 and len(foes) == 1 and spr > 2 and eq > .36 and self._random() < .10:
                return self._finish("raise", bet+cost+int((eligible_pot+cost)*.7), *common)
            if value + .012 >= threshold:
                return self._finish("call", 0, *common)
            return self._finish("fold", 0, *common)
        # Checked to: value, protection and selective continuation/semi-bluffs.
        value_cutoff = .59 if len(foes) == 1 else .46 if len(foes) == 2 else .39
        if eq > value_cutoff and (category >= 1 or len(board) == 5):
            fraction = .70 if texture > .4 else .48
            if len(board) == 5 and eq > .82:
                fraction = .85
            target = bet + max(bb, int(eligible_pot*fraction))
            if spr < 1.2 and eq > .68:
                target = stack+bet
            return self._finish("raise", target, *common)
        fold_rate = sum(self.models.get(p["seat"], _DEFAULT).fold_rate for p in foes) / len(foes)
        initiative = self.last_raiser == seat
        bluff_chance = 0.
        if len(foes) == 1 and spr > 1.7:
            if draw > .4:
                bluff_chance = .52 if in_position else .25
            elif len(board) == 3 and initiative:
                bluff_chance = .48 if texture < .5 else .20
            elif in_position and fold_rate > .55:
                bluff_chance = min(.35, (fold_rate-.45)*1.7)
        if self._random() < bluff_chance:
            return self._finish("raise", bet+max(bb, int(eligible_pot*.42)), *common)
        return self._finish("check", 0, *common)

    def _start_hand(self, hand):
        self.hand_id = hand
        self.roles.clear()
        self.preflop_raises = 0
        self.last_raiser = None

    def _range(self, seat, n, preflop=True):
        m = self.models.get(seat, _DEFAULT)
        role = self.roles.get(seat, "")
        delta = (m.vpip-.25) * .7
        if role == "reraise":
            return max(.05, min(.2, .075 + delta*.35))
        if role == "raise":
            return max(.08, min(.6, (.22 if n > 4 else .38) + delta))
        if role == "call":
            return max(.16, min(.85, .44 + delta))
        return max(.20, min(.95, .55 + delta))

    def _preflop(self, obs, hole, k, foes, behind, n, ip, eff, bb, pot, threshold, icm, common,layers=None):
        legal, stack, bet, cost = common
        ranks = [c % 13 for c in hole]
        pair = ranks[0] == ranks[1]
        high, low = max(ranks), min(ranks)
        suited = hole[0] // 13 == hole[1] // 13
        pct = OPEN_PERCENTILE[k]
        max_bet = max(int(p.get("bet", 0)) for p in foes)
        unopened = max_bet <= bb and self.preflop_raises == 0
        # With no history, wager size still separates an open from a re-raise.
        raise_count = self.preflop_raises or (1 if max_bet > bb else 0)
        caller_fraction = .18 if n >= 5 else .27 if n >= 3 else .40
        aggressor = self.last_raiser
        if aggressor is not None and aggressor != obs["hero"]:
            caller_fraction = self._range(aggressor, n)
        if raise_count >= 2 or max_bet >= 9*bb:
            caller_fraction = min(caller_fraction, .09 if max_bet < 25*bb else .055)
        if max_bet >= 5*bb and eff <= 15:
            caller_fraction = .18 if n >= 5 else .28 if n >= 3 else .48
            if aggressor is not None and aggressor != obs["hero"] and aggressor in self.models:
                model = self.models[aggressor]
                pfr = model.raises / model.hands
                caller_fraction = max(.06,min(.8,caller_fraction+(pfr-1/6)*.8))
        # Only foes already investing voluntarily should count as callers when
        # assessing an open raise. Folded blinds are not showdown opponents.
        invested = [p for p in foes if p.get("bet", 0) > bb]
        showdown_n = max(1, len(invested))
        eq = preflop_equity(k, caller_fraction, showdown_n)
        if layers:
            eq,_=self.equity.estimate(hole,[],[caller_fraction]*len(foes),layers=layers)
        self.last_info = {"street": 0, "equity": eq, "threshold": threshold, "range": caller_fraction, "open_percentile": pct, "icm": icm}
        if unopened:
            if cost == 0 and pct > .45:
                return self._finish("check", 0, *common)
            # Position-sensitive opening frequencies. players_to_act is the
            # number of non-all-in opponents still able to act this round.
            opening = (.54, .48, .35, .28, .23, .19, .16, .14)[min(7, behind)]
            if n == 2:
                opening = .78 if bet else .66
            limpers = sum(bb <= int(p.get("bet", 0)) <= bb and p["seat"] != obs.get("big_blind_seat") for p in foes)
            limpers = max(0, limpers - (1 if obs.get("big_blind_seat") is None else 0))
            if eff <= 12:
                call_frac = max(.10, min(.36, .11 + (12-eff)*.009 + (n <= 3)*.09))
                if foes:
                    willingness = sum(self.models.get(p["seat"],_DEFAULT).vpip for p in foes)/len(foes)
                    call_frac = max(.07,min(.65,call_frac+(willingness-.25)*.48))
                shove_eq = preflop_equity(k, call_frac)
                opponents_to_fold = max(1, behind)
                fold_prob = (1-call_frac) ** opponents_to_fold
                risk = min(stack, max(int(p["stack"])+int(p.get("bet", 0))-bet for p in foes))
                risk = max(0,risk)
                opp_add = max(0, risk+bet-max_bet)
                ev = fold_prob*pot
                for callers in range(1,opponents_to_fold+1):
                    probability = math.comb(opponents_to_fold,callers)*call_frac**callers*(1-call_frac)**(opponents_to_fold-callers)
                    equity = shove_eq if callers==1 else preflop_equity(k,call_frac,callers)
                    ev += probability*(equity*(pot+risk+callers*opp_add)-risk)
                # A blind-vs-blind terminal shove has a tractable exact ICM
                # utility tree even when folded players remain at the table.
                # Multiway shove decisions retain the chip-EV approximation.
                push_good = ev > bb*.04
                payouts = obs.get("payouts")
                if len(foes)==1 and payouts is not None:
                    alive=[p for p in obs["players"] if p["stack"]>0 or p.get("committed",0)>0]
                    if 1<len(alive)<=8 and len(payouts)>=len(alive) and all(payouts[i]>=payouts[i+1]>=0 for i in range(len(payouts)-1)):
                        hi=next(i for i,p in enumerate(alive) if p["seat"]==obs["hero"])
                        vi=next(i for i,p in enumerate(alive) if p["seat"]==foes[0]["seat"])
                        stacks=[int(p["stack"]) for p in alive]
                        folded=list(stacks);folded[vi]+=pot
                        stolen=list(stacks);stolen[hi]+=pot
                        won=list(stacks);won[hi]+=pot+opp_add;won[vi]-=opp_add
                        lost=list(stacks);lost[hi]-=risk;lost[vi]+=pot+risk
                        uf=hero_icm(folded,hi,payouts)
                        utility=fold_prob*hero_icm(stolen,hi,payouts)+(1-fold_prob)*(shove_eq*hero_icm(won,hi,payouts)+(1-shove_eq)*hero_icm(lost,hi,payouts))
                        push_good=utility>uf+1e-10
                        self.last_info["shove_icm"]=True
                if push_good and (pct < min(.9, opening*1.5) or eff < 3):
                    return self._finish("raise", stack+bet, *common)
                if cost == 0:
                    return self._finish("check", 0, *common)
                # Completing the small blind at very short stacks creates
                # poor realization; the policy either shoves or folds.
                return self._finish("fold", 0, *common)
            if pct <= opening - min(.08, limpers*.018):
                size = (2.25 if ip else 2.7) + limpers*.85
                return self._finish("raise", int(size*bb), *common)
            # Cheap multiway set mining/suited speculative entries, only with
            # depth and good immediate price; never open limp by default.
            if limpers and cost and eff > 25 and (pair or suited and high-low <= 3) and cost <= pot*.15:
                return self._finish("call", 0, *common)
            return self._finish("check" if not cost else "fold", 0, *common)
        terminal = cost == stack or all(p["stack"] == 0 for p in foes)
        if terminal or cost >= stack*.55:
            return self._finish("call" if eq >= threshold+.008 else "fold", 0, *common)
        premium = pair and high >= 10 or high == 12 and low == 11
        wide_value = pair and high >= 9 and caller_fraction > .26 or high == 12 and low == 10 and suited and caller_fraction > .3
        if premium or wide_value:
            if eff < 23 or max_bet > stack*.22 or raise_count >= 3:
                if eq > .49 or pair and high >= 11:
                    return self._finish("raise", stack+bet, *common)
            elif raise_count <= 1 or pair and high >= 11:
                return self._finish("raise", int(max_bet*(3.1 if ip else 3.8)), *common)
        realization = .90 if ip else .77
        if pair:
            realization += .04
        if suited:
            realization += .04
        if showdown_n > 1:
            realization -= .08
        required = threshold + (.025 if raise_count >= 2 else .012)
        # Deep implied-odds calls remain capped to a small fraction of stack.
        setmine = pair and eff > 30 and cost < stack*.055 and cost < pot*.7
        if eq * realization >= required and (pct < .58 or cost <= bb*1.2) or setmine:
            return self._finish("call", 0, *common)
        # Occasional blocker three-bets in late heads-up confrontations.
        if raise_count == 1 and high == 12 and suited and low <= 3 and n <= 4 and eff > 35 and self._random() < .14:
            return self._finish("raise", int(max_bet*(3.1 if ip else 3.8)), *common)
        return self._finish("fold", 0, *common)

    @staticmethod
    def _pot_layers(players,me,foes,cost,players_to_act):
        """Normalized, known payout pools for a closed response to an all-in.

        Unknown future calls are not treated as free money. If other players
        still owe actions, retain the conservative full-field approximation.
        """
        if not any(p["stack"]==0 for p in foes) or not all("committed" in p for p in players):
            return None
        cap=int(me["committed"])+cost
        if cap<=0:
            return None
        if players_to_act!=0 and any(p["stack"]>0 and p["committed"]<cap for p in foes):
            return None
        contributions=[min(cap,int(p["committed"])+(cost if p["seat"]==me["seat"] else 0)) for p in players]
        levels=sorted(set(x for x in contributions if x>0))
        grouped={}
        previous=0
        for level in levels:
            amount=(level-previous)*sum(x>=level for x in contributions)
            mask=sum(1<<j for j,p in enumerate(foes) if p["committed"]>=level)
            grouped[mask]=grouped.get(mask,0)+amount
            previous=level
        if len(grouped)<=1:
            return None
        total=sum(grouped.values())
        return [(mask,amount/total) for mask,amount in grouped.items()]

    def _random(self):
        self.equity.seed = (1664525*self.equity.seed+1013904223) & 0xffffffff
        return self.equity.seed / 4294967296.

    @staticmethod
    def _finish(want, amount, legal, stack, bet, cost):
        if want in ("raise", "bet", "all_in"):
            spec = legal.get("raise", legal.get("bet"))
            name = "raise" if "raise" in legal else "bet"
            if spec is not None:
                lo, hi = int(spec.get("min", 0)), min(stack+bet, int(spec.get("max", stack+bet)))
                if lo > max(bet+cost, 0) and lo <= hi:
                    return {"action": name, "amount": max(lo, min(hi, int(amount)))}
            if "all_in" in legal and amount >= stack+bet:
                return {"action": "all_in", "amount": stack+bet}
            want = "call" if cost else "check"
        if want == "call":
            if not cost and "check" in legal:
                return {"action": "check"}
            if "call" in legal:
                return {"action": "call", "amount": min(stack, int(legal["call"].get("amount", cost)))}
        if want == "fold" and cost and "fold" in legal:
            return {"action": "fold"}
        if "check" in legal:
            return {"action": "check"}
        if "fold" in legal:
            return {"action": "fold"}
        if "call" in legal:
            return {"action": "call", "amount": min(stack, cost)}
        # A malformed empty legal set is outside the documented protocol.
        return {"action": "fold"}


_DEFAULT = Opponent()
