"""Decision-interface, constrained-action and independently priced situations."""
import random
import unittest

from agent import Agent
from baselines import CallingStation, RandomLegal
from simulator import HoldemHand


def cards(text):
    return [4 * "23456789TJQKA".index(c[0]) + "cdhs".index(c[1]) for c in text.split()]


def scenario(hole="Ac Ad", board="", *, stacks=(1000, 1000),
             contributions=(0, 20), street_contributions=None,
             current_bet=20, legal=None, history=(), payouts=None):
    board = cards(board)
    bets = tuple(street_contributions if street_contributions is not None else contributions)
    n = len(stacks)
    call = min(stacks[0], max(0, current_bet - bets[0]))
    obs = dict(hole=cards(hole), board=board, hero=0, button=0,
               street=("preflop" if not board else "flop" if len(board)==3 else "turn" if len(board)==4 else "river"),
               street_index=(0 if not board else len(board)-2),
               stacks=list(stacks), contributions=list(contributions),
               street_contributions=list(bets), in_hand=[True] * n,
               all_in=[s == 0 for s in stacks], pot=sum(contributions),
               current_bet=current_bet, min_raise_to=current_bet+20,
               max_raise_to=stacks[0]+bets[0], to_call=call,
               legal_actions=list(legal if legal is not None else (("fold", "call", "raise") if call else ("check", "raise"))),
               history=list(history), small_blind=10, big_blind=20, hand_number=0,
               active_players=n)
    if payouts is not None:
        obs["payouts"] = payouts
    return obs


class AgentTests(unittest.TestCase):
    def test_premium_opens_and_trash_folds_early_position(self):
        h = HoldemHand([2000] * 8, 0)
        premium, trash = h.observation(), h.observation()
        premium["hole"] = cards("Ac Ad")
        trash["hole"] = cards("7c 2d")
        self.assertEqual(Agent().act(premium)["action"], "raise")
        self.assertEqual(Agent().act(trash)["action"], "fold")

    def test_premium_calls_large_preflop_allin_and_trash_folds(self):
        history = ({"seat":1,"street":"preflop","action":"raise","amount":100,"paid":100},)
        premium = scenario("Ac Ad", stacks=(100,0), contributions=(0,100),
                           current_bet=100, legal=("fold","call"), history=history)
        trash = dict(premium, hole=cards("7c 2d"))
        self.assertEqual(Agent().act(premium)["action"], "call")
        self.assertEqual(Agent().act(trash)["action"], "fold")

    def test_guaranteed_royal_board_chop_is_never_folded(self):
        for payouts in (None, (.7,.3)):
            obs = scenario("2c 3d", "Th Jh Qh Kh Ah", stacks=(1000,0),
                           contributions=(10,1010), street_contributions=(0,1000),
                           current_bet=1000, legal=("fold","call"), payouts=payouts,
                           history=({"seat":1,"street":"river","action":"raise","amount":1000,"paid":1000},))
            self.assertEqual(Agent().act(obs), {"action":"call"})

    def test_shared_board_nuts_are_never_raised(self):
        obs = scenario("2c 3d", "Th Jh Qh Kh Ah", stacks=(1000,1000),
                       contributions=(100,100), street_contributions=(0,0), current_bet=0)
        self.assertEqual(Agent().act(obs), {"action":"check"})

    def test_other_provably_locked_board_nuts_call_large_price(self):
        # Quads with the highest possible shared kicker and rainbow broadway
        # cannot be improved by any opponent's private cards.
        for board in ("Ac Ad Ah As Kc", "Kc Kd Kh Ks Ac", "Tc Jd Qh Ks Ac"):
            obs = scenario("2c 3d",board,stacks=(1000,0,0),
                           contributions=(10,1010,1010),street_contributions=(0,1000,1000),
                           current_bet=1000,legal=("fold","call"),payouts=(.5,.3,.2),
                           history=({"seat":1,"street":"river","action":"raise","amount":1000,"paid":1000},
                                    {"seat":2,"street":"river","action":"call","amount":1000,"paid":1000}))
            self.assertEqual(Agent().act(obs), {"action":"call"})

    def test_price_excludes_inaccessible_sidepot_and_uncalled_overbet(self):
        obs = scenario("Ac Ad", stacks=(100,1000,3000), contributions=(100,1000,200),
                       street_contributions=(0,900,100), current_bet=900, legal=("fold","call"))
        obs["in_hand"][2] = False
        # Hero's final contribution is 200: existing contestable chips are
        # own100 + villain200 + folded200 =500; call100 buys share of600.
        bot = Agent()
        self.assertEqual(bot._eligible_pot(obs, 100), 500)
        self.assertAlmostEqual(bot._threshold(obs, 100, [1]), 1/6)
        self.assertNotAlmostEqual(bot._threshold(obs, 100, [1]), 100/1400)

    def test_incomplete_allin_raise_and_round_total_bounds(self):
        bot = Agent()
        obs = scenario(stacks=(15,100), contributions=(10,20), current_bet=20,
                       legal=("fold","call","raise"))
        self.assertEqual(bot._legal(obs, {"action":"raise","amount":1000}),
                         {"action":"raise","amount":25})
        obs = scenario(stacks=(100,100), contributions=(10,20), current_bet=20)
        self.assertEqual(bot._legal(obs, {"action":"raise","amount":21}),
                         {"action":"raise","amount":40})
        self.assertEqual(bot._legal(obs, {"action":"raise","amount":1000}),
                         {"action":"raise","amount":110})

    def test_raise_not_reopened_never_emits_raise(self):
        h = HoldemHand([1000,1000,1000,140], 3)
        h.step({"action":"raise","amount":100})
        h.step({"action":"raise","amount":140})
        h.step({"action":"call"})
        h.step({"action":"call"})
        obs = h.observation()
        obs["hole"] = cards("Ac Ad")
        action = Agent().act(obs)
        self.assertIn(action["action"], ("fold","call"))
        h.step(action)

    def test_missing_data_fallback_is_check_or_fold(self):
        bot = Agent()
        self.assertEqual(bot.act({"legal_actions":["check"]}), {"action":"check"})
        self.assertEqual(bot.act({"legal_actions":["fold","call"]}), {"action":"fold"})
        self.assertEqual(bot.fallbacks, 2)

    def test_history_is_counted_once_and_reset_clears_table(self):
        h = HoldemHand([2000] * 3, 0)
        h.step({"action":"raise","amount":60})
        obs = h.observation()
        bot = Agent()
        bot.act(obs)
        first = [s[:] for s in bot.stats]
        bot.act(obs)
        self.assertEqual(bot.stats, first)
        self.assertEqual(bot.stats[0][1:3], [1,1])
        bot.reset()
        self.assertEqual(bot.stats, [])
        self.assertEqual(bot.decisions, 0)

    def test_unseen_later_seats_are_not_counted_as_folding(self):
        h = HoldemHand([2000]*8, 0)
        bot = Agent()
        bot.act(h.observation())
        self.assertTrue(all(row[0] == 0 for row in bot.stats))
        h.step({'action':'raise','amount':60})
        bot.act(h.observation())
        self.assertEqual(bot.stats[3][:3], [1,1,1])
        self.assertEqual(bot.stats[7][0], 0)

    def test_tight_range_mixture_remains_a_probability(self):
        # A rarely observed, very tight preflop raiser once pushed the mixture
        # weight above1, producing negative equity on an exact-zero river.
        bot=Agent()
        bot.stats=[[200,0,0,0,0,0] for _ in range(3)]
        obs=dict(hole=[30,26],board=[23,50,2,17,4],hero=0,button=2,
            street='river',stacks=[2000]*3,contributions=[80]*3,
            street_contributions=[0]*3,in_hand=[True]*3,all_in=[False]*3,
            pot=240,current_bet=0,min_raise_to=20,max_raise_to=2000,to_call=0,
            legal_actions=['check','raise'],small_blind=10,big_blind=20,
            hand_number=1,history=[dict(seat=2,street='preflop',action='raise',amount=80,paid=80)])
        action=bot.act(obs)
        self.assertIn(action['action'],obs['legal_actions'])
        self.assertEqual(bot.fallbacks,0)

    def test_synthetic_random_states_emit_legal_actions(self):
        rng = random.Random(129912)
        bot = Agent()
        checked = 0
        for seed in range(150):
            n = rng.randint(2,8)
            h = HoldemHand([rng.randint(1,2000) for _ in range(n)], rng.randrange(n),
                           5,10,rng=rng,hand_number=seed)
            rivals = [RandomLegal(seed * 8 + i) for i in range(n)]
            while not h.done:
                if h.to_act == 0:
                    action = bot.act(h.observation())
                    self.assertIn(action["action"], h.legal_actions())
                    h.step(action)
                    checked += 1
                else:
                    h.step(rivals[h.to_act].act(h.observation()))
        self.assertGreater(checked, 100)
        self.assertEqual(bot.fallbacks, 0)


if __name__ == "__main__":
    unittest.main()
