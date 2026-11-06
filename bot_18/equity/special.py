"""Exact rare-board shortcuts that avoid seven-card enumeration at decision time."""
from bisect import bisect_left
from math import comb
from .preflop import CLASS_ORDER, class_index
from .evaluator import STRAIGHT


_buckets=[[] for _ in range(169)]
for _a in range(52):
    for _b in range(_a+1,52):
        _buckets[class_index((_a,_b))].append(((1<<_a)|(1<<_b),max(_a,_b)//4+2,(1<<(_a//4))|(1<<(_b//4))))
_COMBOS=[];_ENDS=[]
for _cls in CLASS_ORDER:
    _COMBOS.extend(_buckets[_cls]);_ENDS.append(len(_COMBOS))
del _buckets


def _choose(n,k):
    return comb(n,k) if 0<=k<=n else 0


def _uniform_kicker_share(lower,equal,higher,opponents,board_playing=False):
    """Exact combinatorial share from equally likely unordered opponent cards.

If k equal-kicker cards are dealt, count the occupied two-card opponent hands.
The count for j occupied hands is C(n,j) C(j,k-j) 2**(2j-k).
"""
    m=2*opponents;total=_choose(lower+equal+higher,m)
    if board_playing:
        return _choose(lower+equal,m)/total/(opponents+1)
    result=0.0
    for k in range(min(equal,m)+1):
        probability=_choose(equal,k)*_choose(lower,m-k)/total
        if not probability:continue
        arrangements=_choose(m,k);share=0.0
        for j in range((k+1)//2,min(k,opponents)+1):
            ways=_choose(opponents,j)*_choose(j,k-j)*(2**(2*j-k))
            share+=ways/(j+1)
        result+=probability*share/arrangements
    return result


def quads_on_board(hole,board,range_strength,opponents,descriptor):
    """Exact for heads-up arbitrary local prefix range and random-range multiway.

Return None for multiple constrained opponents, handled by the learned model.
The caller has already established descriptor[13] == 7 on a five-card board.
"""
    fraction=max(.06,min(1.0,float(range_strength)))
    opponents=max(1,min(7,int(opponents)))
    if opponents>1 and fraction<1.0:return None
    board_kicker=descriptor[15]
    hero=max(descriptor[1],descriptor[2],board_kicker)
    blocked=0
    for c in (*hole,*board):blocked|=1<<c
    if fraction==1.0:
        lower=equal=higher=0
        for c in range(52):
            if (blocked>>c)&1:continue
            rank=c//4+2
            if rank<hero:lower+=1
            elif rank==hero:equal+=1
            else:higher+=1
        return _uniform_kicker_share(lower,equal,higher,opponents,hero==board_kicker)
    end=_ENDS[bisect_left(_ENDS,fraction*1326)]
    wins=ties=count=0
    for bits,high,_ in _COMBOS[:end]:
        if bits&blocked:continue
        high=max(high,board_kicker);count+=1
        wins+=high<hero;ties+=high==hero
    return (wins+.5*ties)/count


def straight_on_board(hole,board,range_strength,opponents,descriptor):
    """Exact heads-up equity on an unpaired straight board with no possible flush."""
    if int(opponents)!=1 or descriptor[30]>2:return None
    fraction=max(.06,min(1.0,float(range_strength)))
    blocked=board_mask=0
    for c in (*hole,*board):blocked|=1<<c
    for c in board:board_mask|=1<<(c//4)
    hero=descriptor[8]
    end=_ENDS[bisect_left(_ENDS,fraction*1326)]
    wins=ties=count=0
    for bits,_,ranks in _COMBOS[:end]:
        if bits&blocked:continue
        high=STRAIGHT[board_mask|ranks];count+=1
        wins+=high<hero;ties+=high==hero
    return (wins+.5*ties)/count
