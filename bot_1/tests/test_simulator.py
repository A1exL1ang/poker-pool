import unittest
from pokerbot.simulator import Table, run_sng
from pokerbot.agent import Agent


class Script:
    def __init__(self, actions):
        self.actions=list(actions)
        self.observations=[]
    def act(self,obs):
        self.observations.append(obs)
        if self.actions:
            return self.actions.pop(0)
        if any(x["action"]=="check" for x in obs["legal_actions"]):
            return {"action":"check"}
        return {"action":"call","amount":obs["to_call"]}


class SimulatorTests(unittest.TestCase):
    def test_hu_blind_and_action_order(self):
        a,b=Script([]),Script([])
        t=Table([a,b],10)
        t.play_hand()
        self.assertEqual(a.observations[0]["button"],0)
        self.assertEqual(a.observations[0]["hero"],0)
        self.assertEqual(a.observations[0]["to_call"],10)
        self.assertEqual(b.observations[1]["board"].__len__(),3)
        t.check_chips()

    def test_uncalled_allin_refund(self):
        a=Script([{"action":"raise","amount":2000}])
        b=Script([{"action":"fold"}])
        t=Table([a,b],10)
        t.play_hand()
        self.assertEqual([p.stack for p in t.players],[2020,1980])
        self.assertEqual(t.showdowns,0)

    def test_layered_side_pots(self):
        t=Table([Script([]) for _ in range(4)],1)
        values=[(0,100,False),(0,200,False),(0,300,False),(0,300,True)]
        for p,(stack,commit,fold) in zip(t.players,values):
            p.stack,p.committed,p.folded=stack,commit,fold
        t.award_side_pots({0:100,1:90,2:80})
        # Main 400 to seat0; next 300 to seat1; final 200 to seat2.
        self.assertEqual([p.stack for p in t.players],[400,300,200,0])

    def test_odd_chip_left_of_button(self):
        t=Table([Script([]) for _ in range(3)],1)
        t.button=0
        for p in t.players:
            p.stack=0
            p.committed=1
        t.players[2].folded=True
        t.award_side_pots({0:10,1:10})
        self.assertEqual([p.stack for p in t.players],[1,2,0])

    def test_short_allin_does_not_reopen(self):
        # 3 players: dealer 0 opens to 100, SB calls, BB shoves to 130.
        # The original raiser must call/fold because the extra 30 < full 80.
        a=Script([{"action":"raise","amount":100},{"action":"call","amount":30}])
        b=Script([{"action":"call","amount":90},{"action":"call","amount":30}])
        c=Script([{"action":"raise","amount":130}])
        t=Table([a,b,c],1)
        t.players[2].stack=130
        t.total=4130
        t.play_hand()
        self.assertNotIn("raise",[x["action"] for x in a.observations[1]["legal_actions"]])
        self.assertNotIn("raise",[x["action"] for x in b.observations[1]["legal_actions"]])
        t.check_chips()

    def test_cumulative_short_allins_reopen(self):
        # 4 players: UTG3 raises to100, dealer0 calls, SB1 to140,
        # BB2 to180. Cumulative 80 reopens UTG's full-raise right.
        ps=[Script([{"action":"call","amount":100},{"action":"call","amount":80}]),
            Script([{"action":"raise","amount":140}]),
            Script([{"action":"raise","amount":180}]),
            Script([{"action":"raise","amount":100},{"action":"call","amount":80}])]
        t=Table(ps,1)
        t.players[1].stack=140
        t.players[2].stack=180
        t.total=4320
        t.play_hand()
        self.assertIn("raise",[x["action"] for x in ps[3].observations[1]["legal_actions"]])
        t.check_chips()

    def test_agents_full_tournaments(self):
        for seed in range(4):
            t,points=run_sng([Agent(seed*8+i) for i in range(8)],seed+501)
            self.assertEqual(t.illegal,0)
            self.assertAlmostEqual(sum(points),100)
            self.assertEqual(sum(p.stack for p in t.players),16000)


if __name__=="__main__":
    unittest.main()
