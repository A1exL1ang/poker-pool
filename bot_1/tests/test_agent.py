import copy
import random
import unittest
from unittest.mock import patch
from pokerbot.agent import Agent
from pokerbot.icm import hero_icm, call_threshold


def observation(hole=None, board=None, players=2, cost=10):
    ps = [{"seat":i, "stack":1980, "bet":20, "committed":20} for i in range(players)]
    ps[0].update(stack=1980+cost, bet=20-cost, committed=20-cost)
    return {"table_id":"test", "hand_id":1, "hero":0, "button":players-1,
            "hole_cards":hole or [12, 25], "board":board or [], "big_blind":20,
            "pot":players*20-cost, "players":ps, "players_to_act":players-1,
            "legal_actions":[{"action":"fold"}, {"action":"call", "amount":cost},
                             {"action":"raise", "min":40, "max":2000}] if cost else
                            [{"action":"check"}, {"action":"raise", "min":40,"max":2000}]}


class PolicyTests(unittest.TestCase):
    def test_aa_opens(self):
        self.assertEqual(Agent().act(observation())["action"], "raise")

    def test_72o_utg_folds(self):
        obs = observation([5,13], players=8, cost=20)
        obs["players"][1]["bet"] = 10
        for p in obs["players"][2:7]:
            p["bet"] = 0
        self.assertEqual(Agent().act(obs)["action"], "fold")

    def test_no_folding_free_checks(self):
        obs = observation([5,13], players=8, cost=0)
        self.assertIn(Agent().act(obs)["action"], ("check","raise"))

    def test_no_raise_when_not_legal(self):
        obs = observation()
        obs["legal_actions"] = [{"action":"fold"},{"action":"call","amount":10}]
        self.assertEqual(Agent().act(obs), {"action":"call","amount":10})

    def test_short_allin_raise(self):
        obs = observation()
        obs["players"][0].update(stack=20, bet=10)
        obs["legal_actions"][-1].update(min=30,max=30)
        self.assertEqual(Agent().act(obs), {"action":"raise","amount":30})

    def test_raise_clamp(self):
        a = Agent._finish("raise", 100000, {"raise":{"min":50,"max":100},"call":{}}, 90,10,20)
        self.assertEqual(a, {"action":"raise","amount":100})

    def test_guaranteed_board_chop_call(self):
        obs = observation([0, 14], [47,48,49,50,51], cost=10)
        self.assertEqual(Agent().act(obs)["action"], "call")

    def test_side_pot_layer_weights(self):
        ps=[{"seat":0,"stack":100,"committed":100},
            {"seat":1,"stack":0,"committed":50},
            {"seat":2,"stack":0,"committed":200}]
        layers=Agent._pot_layers(ps,ps[0],ps[1:],100,0)
        self.assertEqual(layers,[(3,1/3),(2,2/3)])
        self.assertAlmostEqual(sum(w for m,w in layers),1.)

    def test_side_pot_future_caller_is_not_free_money(self):
        ps=[{"seat":0,"stack":100,"committed":100},
            {"seat":1,"stack":0,"committed":50},
            {"seat":2,"stack":1000,"committed":80}]
        self.assertIsNone(Agent._pot_layers(ps,ps[0],ps[1:],100,1))

    def test_royal_board_side_pot_chop(self):
        obs=observation([0,14],[47,48,49,50,51],players=3,cost=0)
        obs["players"]=[{"seat":0,"stack":100,"bet":0,"committed":100},
                        {"seat":1,"stack":0,"bet":0,"committed":50},
                        {"seat":2,"stack":0,"bet":100,"committed":200}]
        obs["pot"]=350
        obs["to_call"]=100
        obs["players_to_act"]=0
        obs["legal_actions"]=[{"action":"fold"},{"action":"call","amount":100}]
        bot=Agent()
        self.assertEqual(bot.act(obs)["action"],"call")
        self.assertAlmostEqual(bot.last_info["equity"],4/9)
        self.assertEqual(bot.last_info["side_pot_layers"],2)

    def test_reset_and_events(self):
        bot=Agent()
        obs=observation()
        event={"id":1,"seat":1,"hand_id":1,"street":"preflop","action":"raise"}
        obs["history"]=[event]
        bot.act(obs)
        hands=bot.models[1].hands
        bot.act(obs)
        self.assertEqual(bot.models[1].hands,hands)
        obs["table_id"]="another"
        obs["history"]=[]
        bot.act(obs)
        self.assertFalse(bot.models)

    def test_external_event_feed_preserves_roles(self):
        bot=Agent()
        bot.reset("test")
        event={"id":1,"seat":1,"hand_id":1,"street":"preflop","action":"raise"}
        bot.observe([event])
        self.assertEqual(bot.last_raiser,1)
        bot.act(observation())
        self.assertEqual(bot.last_raiser,1)
        bot.observe([dict(event,id=2,hand_id=2,seat=0)])
        self.assertEqual(bot.last_raiser,0)
        self.assertEqual(bot.preflop_raises,1)

    def test_malformed_input_fails_closed(self):
        for obs in (None,{}, observation([12,12]), observation([99,0])):
            action=Agent().act(obs)
            self.assertIn(action["action"], ("fold","check"))

    def test_no_gameplay_network_or_disk(self):
        bot=Agent()
        obs=observation([12,25],[4,18,33],cost=20)
        with patch("builtins.open",side_effect=AssertionError("gameplay file I/O")),patch("socket.socket",side_effect=AssertionError("gameplay network")),patch("subprocess.Popen",side_effect=AssertionError("gameplay process")):
            result=bot.act(obs)
        self.assertIn(result["action"],("fold","call","raise"))

    def test_fuzz_legal_actions(self):
        rng=random.Random(5301)
        a=Agent()
        for i in range(3000):
            n=rng.randint(2,8)
            street=rng.choice((0,3,4,5))
            cards=rng.sample(range(52),street+2)
            cost=rng.randrange(0,200)
            obs=observation(cards[:2],cards[2:],n,cost=min(20,cost))
            obs["hand_id"]=i
            obs["pot"]=rng.randint(40,4000)
            stack=rng.randint(1,2000)
            obs["players"][0]["stack"]=stack
            cost=min(stack,cost)
            obs["to_call"]=cost
            legal=[{"action":"fold"},{"action":"call","amount":cost}] if cost else [{"action":"check"}]
            if rng.random()<.7 and stack>cost+20:
                lo=obs["players"][0]["bet"]+cost+20
                hi=obs["players"][0]["bet"]+stack
                legal.append({"action":"raise","min":lo,"max":hi})
            obs["legal_actions"]=legal
            result=a.act(obs)
            allowed={x["action"]:x for x in legal}
            self.assertIn(result["action"],allowed)
            if result["action"]=="raise":
                spec=allowed["raise"]
                self.assertLessEqual(result["amount"],spec["max"])
                self.assertGreaterEqual(result["amount"],spec["min"])
            elif result["action"]=="call":
                self.assertEqual(result["amount"],cost)


class ICMTests(unittest.TestCase):
    def test_symmetry(self):
        for i in range(8):
            self.assertAlmostEqual(hero_icm([2000]*8,i,[50,30,20,0,0,0,0,0]),12.5)

    def test_winner_take_all_is_chip_fraction(self):
        s=[30,20,10,40]
        for i in range(4):
            self.assertAlmostEqual(hero_icm(s,i,[100,0,0,0]),s[i])
        self.assertAlmostEqual(call_threshold([80,70],0,1,50,20,[100,0]),20/70)

    def test_payout_sum(self):
        s=[100,200,300,400]
        p=[50,30,20,0]
        self.assertAlmostEqual(sum(hero_icm(s,i,p) for i in range(4)),100)

    def test_bubble_risk_premium(self):
        threshold=call_threshold([100,80,10],0,1,30,80,[50,30,0])
        self.assertGreater(threshold,80/110)

    def test_busted(self):
        self.assertEqual(hero_icm([0,100,200],0,[50,30,20]),20)


if __name__=="__main__":
    unittest.main()
