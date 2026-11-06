"""Original local NLHE validation engine; not the official contest simulator.

Implements all-in runouts, layered side pots, odd chips, uncalled bet refunds,
heads-up blind positions, and raise reopening after cumulative short all-ins.
The CLI's schedules/payouts are synthetic stress fixtures, never contest facts.
"""
from dataclasses import dataclass
import random
from .evaluator import evaluate
from .icm import hero_icm


@dataclass
class Player:
    seat: int
    stack: int
    bet: int = 0
    committed: int = 0
    folded: bool = False


class Table:
    def __init__(self, policies, seed=1, starting_stack=2000, strict=True):
        self.policies=policies
        self.players=[Player(i,starting_stack) for i in range(len(policies))]
        self.rng=random.Random(seed)
        self.button=-1
        self.hand_id=0
        self.table_id=f"local-{seed}"
        self.event_id=0
        self.history=[]
        self.illegal=0
        self.strict=strict
        self.total=starting_stack*len(policies)
        self.fold_wins=0
        self.showdowns=0
        self.decisions=0
        self.eliminations=[]
        for policy in policies:
            if hasattr(policy,"reset"):
                policy.reset(self.table_id)

    def _next(self, seat, seats):
        return min(seats,key=lambda s:(s-seat-1)%len(self.players))

    def _order(self, first, seats):
        return sorted(seats,key=lambda s:(s-first)%len(self.players))

    def _put(self, p, amount):
        amount=min(p.stack,max(0,int(amount)))
        p.stack-=amount
        p.bet+=amount
        p.committed+=amount

    @property
    def pot(self):
        return sum(p.committed for p in self.players)

    def check_chips(self):
        assert all(p.stack>=0 and p.bet>=0 and p.committed>=p.bet for p in self.players)
        assert sum(p.stack+p.committed for p in self.players)==self.total

    def play_hand(self, big_blind=20, payouts=None):
        seats=[p.seat for p in self.players if p.stack>0]
        if len(seats)<2:
            return False
        self.hand_id+=1
        self.button=self._next(self.button,seats)
        self.history=[]
        for p in self.players:
            p.bet=p.committed=0
            p.folded=p.seat not in seats
        sb=self.button if len(seats)==2 else self._next(self.button,seats)
        bb=self._next(sb,seats)
        self.sb,self.bb=sb,bb
        self.big_blind=big_blind
        self._put(self.players[sb],max(1,big_blind//2))
        self._put(self.players[bb],big_blind)
        deck=list(range(52))
        self.rng.shuffle(deck)
        self.holes={s:[deck.pop(),deck.pop()] for s in seats}
        self.board=[]
        before={p.seat:p.stack+p.committed for p in self.players}
        self._street("preflop", self._next(bb,seats), payouts)
        for street,count in (("flop",3),("turn",1),("river",1)):
            if len([p for p in self.players if not p.folded])<=1:
                break
            for p in self.players:
                p.bet=0
            self.board.extend(deck.pop() for _ in range(count))
            self._street(street,self._next(self.button,seats),payouts)
        contenders=[p for p in self.players if not p.folded]
        if len(contenders)==1:
            contenders[0].stack+=self.pot
            self.fold_wins+=1
        else:
            scores={p.seat:evaluate(self.holes[p.seat]+self.board) for p in contenders}
            self.award_side_pots(scores)
            self.showdowns+=1
        for p in self.players:
            p.committed=p.bet=0
        busted=[p.seat for p in self.players if before[p.seat]>0 and p.stack==0]
        # Simultaneous eliminations rank larger starting stacks higher; exact
        # ties split positional points in the tournament driver.
        busted.sort(key=lambda s:before[s])
        if busted:
            self.eliminations.append([(s,before[s]) for s in busted])
        self.check_chips()
        return True

    def _street(self, street, first, payouts):
        current=max(p.bet for p in self.players)
        increment=self.big_blind
        acted_at={}
        pending=self._order(first,[p.seat for p in self.players if not p.folded and p.stack>0])
        actions=0
        while pending:
            actions+=1
            if actions>300:
                raise RuntimeError("betting round did not terminate")
            if len([p for p in self.players if not p.folded])<2:
                break
            s=pending.pop(0)
            p=self.players[s]
            if p.folded or not p.stack:
                continue
            to_call=min(p.stack,current-p.bet)
            opponents_can_act=any(q.stack>0 and not q.folded and q.seat!=s for q in self.players)
            if not opponents_can_act and to_call==0:
                continue
            legal=[{"action":"fold"},{"action":"call","amount":to_call}] if to_call else [{"action":"check"}]
            can_raise=s not in acted_at or current-acted_at[s]>=increment
            if p.stack>to_call and opponents_can_act and can_raise:
                hi=p.stack+p.bet
                legal.append({"action":"raise","min":min(hi,current+increment),"max":hi})
            live=[q for q in self.players if not q.folded]
            postorder=self._order((self.button+1)%len(self.players),[q.seat for q in live])
            behind=sum(q.stack>0 and q.seat!=s for q in self.players if not q.folded and q.seat in pending)
            obs={"table_id":self.table_id,"hand_id":self.hand_id,"hero":s,"button":self.button,
                 "big_blind":self.big_blind,"small_blind_seat":self.sb,"big_blind_seat":self.bb,
                 "hole_cards":self.holes[s],"board":self.board[:],"pot":self.pot,
                 "players":[{"seat":q.seat,"stack":q.stack,"bet":q.bet,"committed":q.committed,"folded":q.folded} for q in self.players if q.stack>0 or q.committed>0],
                 "players_to_act":behind,"to_call":to_call,"legal_actions":legal,"history":self.history[-16:]}
            if payouts is not None:
                obs["payouts"]=payouts
            result=self.policies[s].act(obs)
            self.decisions+=1
            allowed={x["action"]:x for x in legal}
            name=result.get("action")
            valid=name in allowed
            if valid and name=="raise":
                valid=isinstance(result.get("amount"),int) and allowed[name]["min"]<=result["amount"]<=allowed[name]["max"]
            if valid and name=="call":
                valid=result.get("amount",to_call)==to_call
            if not valid:
                self.illegal+=1
                if self.strict:
                    raise AssertionError(("illegal decision",result,obs))
                name="check" if not to_call else "fold"
            pot_before=self.pot
            oldbet=p.bet
            if name=="fold":
                p.folded=True
            elif name=="call":
                self._put(p,to_call)
            elif name=="raise":
                self._put(p,result["amount"]-p.bet)
                increase=p.bet-current
                if increase>=increment:
                    increment=increase
                current=p.bet
                pending=self._order((s+1)%len(self.players),[q.seat for q in self.players if q.seat!=s and not q.folded and q.stack>0 and q.bet<current])
            acted_at[s]=current
            self.event_id+=1
            self.history.append({"id":self.event_id,"seat":s,"hand_id":self.hand_id,"street":street,
                                 "action":name,"amount":p.bet-oldbet,"pot_before":pot_before,"facing_bet":to_call>0})
            self.check_chips()
        # Return unmatched wagers before a subsequent street or showdown.
        ranked=sorted(self.players,key=lambda p:p.bet,reverse=True)
        if len(ranked)>1 and ranked[0].bet>ranked[1].bet and not ranked[0].folded:
            p=ranked[0]
            refund=p.bet-ranked[1].bet
            p.bet-=refund
            p.committed-=refund
            p.stack+=refund
        self.check_chips()

    def award_side_pots(self, scores):
        levels=sorted(set(p.committed for p in self.players if p.committed>0))
        prev=0
        for level in levels:
            contributors=[p for p in self.players if p.committed>=level]
            amount=(level-prev)*len(contributors)
            eligible=[p.seat for p in contributors if not p.folded]
            assert eligible,"side pot with no eligible winner"
            best=max(scores[s] for s in eligible)
            winners=[s for s in eligible if scores[s]==best]
            share,odd=divmod(amount,len(winners))
            winners=self._order((self.button+1)%len(self.players),winners)
            for i,s in enumerate(winners):
                self.players[s].stack+=share+(i<odd)
            prev=level


def run_sng(policies, seed=1, max_hands=200, schedule=None, payouts=None):
    schedule=schedule or [20,40,80,120,200,400,800,1600]
    payouts=payouts or [50,30,20]+[0]*(len(policies)-3)
    t=Table(policies,seed)
    for h in range(max_hands):
        if not t.play_hand(schedule[min(h//25,len(schedule)-1)],payouts):
            break
    points=[0.]*len(policies)
    remaining=len(policies)
    for group in t.eliminations:
        # Allocate equal-stack simultaneous busts the average of their places.
        i=0
        while i<len(group):
            j=i+1
            while j<len(group) and group[j][1]==group[i][1]:
                j+=1
            award=sum(payouts[remaining-(j-i):remaining])/(j-i)
            for s,_ in group[i:j]:
                points[s]=award
            remaining-=j-i
            i=j
    alive=[p for p in t.players if p.stack>0]
    if len(alive)==1:
        points[alive[0].seat]=payouts[0]
    else:
        for i,p in enumerate(alive):
            points[p.seat]=hero_icm([q.stack for q in alive],i,payouts[:len(alive)])
    assert abs(sum(points)-sum(payouts))<1e-7
    return t,points
