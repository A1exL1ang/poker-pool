from adapters import Adapter, Baseline
from common.engine import Action, Bot, Config, Table, run_tournament

def test_no_adapter_faults_full_tables():
    for seed in (42,43,44):
        bots=[Adapter(i) for i in range(1,9)]
        if seed==43: bots[0]=Baseline()
        r=run_tournament(bots,Config(max_hands=200),seed)
        assert not any(r.errors),r.errors
        assert not any(r.illegal),r.illegal
        assert sum(r.final_stacks)==16000
        assert abs(sum(r.points)-1)<1e-9
        assert all(not getattr(b.bot,'errors',0) for b in bots)

def test_allin_short_blind_call_cost():
    class Capture(Bot):
        def __init__(self): self.seen=[]
        def act(self,o): self.seen.append(o);return Action('call' if o.to_call else 'check')
    bots=[Capture(),Capture()]
    t=Table(bots,Config(n_seats=2,schedule=((10,20,0),),payouts=(1,0)),1)
    # First hand BB is seat zero; the nominal blind is 20 but only 15 posted.
    t.stacks=[15,100]
    t.play_hand()
    assert bots[1].seen[0].to_call==5
    assert sum(t.stacks)==115
