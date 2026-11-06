"""Rule-critical fixtures for the independent local NLHE environment."""

from __future__ import annotations

import json
import random
import time
import unittest

from pokerbot.game import IllegalAction, Table, run_tournament


def passive(observation):
    return {"action": "call" if observation["to_call"] else "check"}


def shove(observation):
    if observation["can_raise"]:
        return {"action": "raise", "amount": observation["max_raise_to"]}
    return passive(observation)


class Script:
    def __init__(self, **streets):
        self.script = {street: list(actions) for street, actions in streets.items()}
        self.seen = []

    def decide(self, observation):
        self.seen.append(observation)
        choices = self.script.get(observation["street"], [])
        action = choices.pop(0) if choices else passive(observation)
        return action(observation) if callable(action) else action


class BettingTests(unittest.TestCase):
    def table(self, stacks, **options):
        return Table(stacks, evaluator=lambda cards: cards[0], **options)

    def test_initial_observation_and_raise_to_semantics(self):
        agents = [Script(preflop=[{"action": "raise", "amount": 60}]), Script(), Script()]
        table = self.table([2000] * 3)
        table.play_hand(agents, deck=range(52))
        first, second, third = (agent.seen[0] for agent in agents)
        self.assertEqual((first["hero"], first["pot"], first["to_call"]), (0, 30, 20))
        self.assertEqual((first["min_raise_to"], first["max_raise_to"]), (40, 2000))
        self.assertEqual((second["pot"], second["to_call"], second["min_raise_to"]), (90, 50, 100))
        self.assertEqual((third["pot"], third["to_call"]), (140, 40))
        self.assertEqual(second["history"][-1]["amount"], 60)
        self.assertEqual(second["history"][-1]["paid"], 60)
        self.assertEqual(third["history"][-1]["amount"], 50)
        self.assertEqual(third["history"][-1]["to"], 60)

    def test_big_blind_gets_option_after_limpers(self):
        agents = [Script(), Script(), Script()]
        result = self.table([500] * 3).play_hand(agents)
        preflop = [h for h in result.history if h["street"] == "preflop" and h["action"] in ("call", "check")]
        self.assertEqual([h["seat"] for h in preflop], [0, 1, 2])
        self.assertEqual(preflop[-1]["action"], "check")
        self.assertTrue(agents[2].seen[0]["can_raise"])

    def test_heads_up_blinds_dealing_and_action_order(self):
        agents = [Script(), Script()]
        result = self.table([200] * 2).play_hand(agents, deck=range(52))
        self.assertEqual((result.button, result.small_blind_seat, result.big_blind_seat), (0, 0, 1))
        self.assertEqual(result.hole_cards, ((1, 3), (0, 2)))
        self.assertEqual(result.board, (5, 6, 7, 9, 11))
        for street, first in (("preflop", 0), ("flop", 1), ("turn", 1), ("river", 1)):
            actions = [h for h in result.history if h["street"] == street and h["action"] in ("check", "call")]
            self.assertEqual(actions[0]["seat"], first)

    def test_short_all_in_does_not_reopen_prior_actors(self):
        agents = [
            Script(preflop=[{"action": "raise", "amount": 40}]),
            Script(),
            Script(preflop=[{"action": "raise", "amount": 45}]),
        ]
        self.table([300, 300, 45]).play_hand(agents)
        returned = [o for o in agents[0].seen if o["street"] == "preflop"][-1]
        self.assertEqual((returned["current_bet"], returned["to_call"]), (45, 5))
        self.assertFalse(returned["can_raise"])
        self.assertEqual(returned["min_raise_to"], 65)

    def test_multiple_short_all_ins_cumulatively_reopen(self):
        agents = [
            Script(),
            Script(preflop=[{"action": "raise", "amount": 80}]),
            Script(preflop=[{"action": "raise", "amount": 100}]),
            Script(preflop=[{"action": "raise", "amount": 60}, {"action": "raise", "amount": 140}]),
        ]
        result = self.table([300, 80, 100, 300]).play_hand(agents)
        returned = [o for o in agents[3].seen if o["street"] == "preflop"][1]
        self.assertEqual((returned["current_bet"], returned["to_call"]), (100, 40))
        self.assertTrue(returned["can_raise"])
        self.assertEqual(returned["min_raise_to"], 140)
        raises = [h for h in result.history if h["action"] == "raise"]
        self.assertEqual([h["full_raise"] for h in raises], [True, False, False, True])

    def test_reopening_is_player_specific(self):
        agents = [
            Script(flop=[{"action": "raise", "amount": 200}]),
            Script(flop=[{"action": "raise", "amount": 100}]),
            Script(flop=[{"action": "raise", "amount": 125}]),
            Script(),
        ]
        self.table([220, 500, 145, 500]).play_hand(agents)
        a = [o for o in agents[1].seen if o["street"] == "flop"][1]
        c = [o for o in agents[3].seen if o["street"] == "flop"][1]
        self.assertTrue(a["can_raise"])
        self.assertEqual(a["to_call"], 100)
        self.assertFalse(c["can_raise"])
        self.assertEqual(c["to_call"], 75)
        self.assertEqual(a["min_raise_to"], 300)
        self.assertEqual(c["min_raise_to"], 300)

    def test_checked_player_cannot_raise_an_opening_short_all_in(self):
        agents = [Script(), Script(), Script(flop=[{"action": "raise", "amount": 5}])]
        self.table([200, 200, 25]).play_hand(agents)
        returned = [o for o in agents[1].seen if o["street"] == "flop"][1]
        self.assertEqual(returned["to_call"], 5)
        self.assertFalse(returned["can_raise"])

    def test_unacted_player_can_raise_after_opening_short_bet(self):
        agents = [
            Script(flop=[{"action": "raise", "amount": 25}]),
            Script(flop=[{"action": "raise", "amount": 5}]),
            Script(),
        ]
        # SB=1 opens for its last 5. BB=2 has not acted and may raise to 25.
        agents[0] = Script()
        agents[2] = Script(flop=[{"action": "raise", "amount": 25}])
        self.table([200, 25, 200]).play_hand(agents)
        initial = [o for o in agents[2].seen if o["street"] == "flop"][0]
        self.assertTrue(initial["can_raise"])
        self.assertEqual(initial["min_raise_to"], 25)

    def test_short_big_blind_does_not_reduce_preflop_bring_in(self):
        agents = [Script(), Script(), Script()]
        result = self.table([200, 200, 5]).play_hand(agents)
        self.assertEqual(agents[0].seen[0]["to_call"], 20)
        self.assertEqual(agents[1].seen[0]["to_call"], 10)
        self.assertEqual(sum(result.ending_stacks), 405)

    def test_no_betting_into_empty_side_pot(self):
        agents = [Script(), Script()]
        result = self.table([100, 5]).play_hand(agents)
        self.assertEqual(sum(len(agent.seen) for agent in agents), 0)
        self.assertEqual(result.returned, (5, 0))
        self.assertEqual(sum(result.ending_stacks), 105)
        self.assertEqual(len(result.board), 5)

    def test_deep_actor_can_only_call_against_all_in_player(self):
        agents = [Script(), Script()]
        result = self.table([5, 100], button=1).play_hand(agents)
        # BTN1 posts 10; BB0 is all-in for 5, so the 5 excess returns.
        self.assertEqual(len(agents[1].seen), 0)
        self.assertEqual(result.returned, (0, 5))

    def test_invalid_underraise_is_rejected(self):
        agents = [Script(preflop=[{"action": "raise", "amount": 30}]), Script(), Script()]
        with self.assertRaisesRegex(IllegalAction, "incomplete raise"):
            self.table([200] * 3).play_hand(agents)

    def test_cannot_raise_when_action_not_reopened(self):
        agents = [
            Script(preflop=[{"action": "raise", "amount": 40}, {"action": "raise", "amount": 100}]),
            Script(), Script(preflop=[{"action": "raise", "amount": 45}]),
        ]
        with self.assertRaisesRegex(IllegalAction, "not open"):
            self.table([200, 200, 45]).play_hand(agents)

    def test_missing_noninteger_and_overstack_raise_rejected(self):
        for amount in (None, 100.0, True, 201, -1):
            with self.subTest(amount=amount):
                agents = [Script(preflop=[{"action": "raise", "amount": amount}]), Script()]
                with self.assertRaises(IllegalAction):
                    self.table([200] * 2).play_hand(agents)


class PotTests(unittest.TestCase):
    def test_actual_cards_showdown_and_board_only_tie(self):
        from pokerbot.interface import card
        def make_deck(holes, board):
            # Three seats, button zero: seat1, seat2, seat0 get each round.
            prefix = [card(holes[seat][round]) for round in range(2) for seat in (1, 2, 0)]
            reserved = set(prefix) | {card(c) for c in board}
            burns = [c for c in range(52) if c not in reserved][:3]
            prefix += [burns[0]] + [card(c) for c in board[:3]] + [burns[1], card(board[3]), burns[2], card(board[4])]
            return prefix + [c for c in range(52) if c not in prefix]
        holes = [("As", "Ad"), ("Ks", "Kd"), ("Qs", "Qd")]
        deck = make_deck(holes, ("2c", "5d", "7h", "9s", "Jc"))
        result = Table([100, 200, 300]).play_hand([shove] * 3, deck=deck)
        self.assertEqual(result.awards, (300, 200, 0))
        # Last-to-act BB only needs to call the 200 maximum that can be won.
        self.assertEqual(result.returned, (0, 0, 0))
        self.assertEqual(result.ending_stacks, (300, 200, 100))
        holes = [("4c", "4d"), ("2c", "2d"), ("3c", "3d")]
        deck = make_deck(holes, ("As", "Ks", "Qs", "Js", "Ts"))
        result = Table([100, 200, 300]).play_hand([shove] * 3, deck=deck)
        self.assertEqual(result.ending_stacks, (100, 200, 300))

    def test_three_side_pots_and_uncalled_overbet(self):
        # Four players receive first hole cards 3,0,1,2 in seat order.
        table = Table([100, 200, 300, 400], evaluator=lambda cards: {0: 40, 1: 30, 2: 20, 3: 10}[cards[0]])
        result = table.play_hand([shove] * 4, deck=range(52))
        self.assertEqual(result.returned, (0, 0, 0, 100))
        self.assertEqual([pot.amount for pot in result.pots], [400, 300, 200])
        self.assertEqual([pot.eligible for pot in result.pots], [(0, 1, 2, 3), (1, 2, 3), (2, 3)])
        self.assertEqual(result.awards, (0, 700, 200, 0))
        self.assertEqual(result.ending_stacks, (0, 700, 200, 100))
        self.assertEqual(result.eliminated, ((0,),))

    def test_folded_contributions_remain_but_folded_hand_cannot_win(self):
        agents = [
            Script(turn=[{"action": "fold"}]),
            Script(flop=[{"action": "raise", "amount": 60}]),
            Script(turn=[{"action": "raise", "amount": 100}]),
        ]
        # Folded BTN0 has largest evaluator score, so exclusion is material.
        table = Table([500] * 3, evaluator=lambda cards: {2: 100, 0: 10, 1: 20}[cards[0]])
        result = table.play_hand(agents, deck=range(52))
        self.assertEqual(sum(p.amount for p in result.pots), 440)
        self.assertEqual(result.awards, (0, 0, 440))
        self.assertTrue(all(0 not in pot.eligible for pot in result.pots))
        self.assertEqual(result.ending_stacks, (420, 320, 760))

    def test_odd_chip_goes_clockwise_left_of_button(self):
        table = Table([1, 1, 1], small_blind=0, big_blind=1, evaluator=lambda cards: {2: 10, 0: 0, 1: 10}[cards[0]])
        result = table.play_hand([passive] * 3, deck=range(52))
        self.assertEqual(result.awards, (1, 0, 2))
        self.assertEqual(result.pots[0].shares, ((2, 2), (0, 1)))

    def test_fold_win_returns_unmatched_part_before_awarding(self):
        agents = [Script(preflop=[{"action": "raise", "amount": 200}]), lambda o: {"action": "fold"}, lambda o: {"action": "fold"}]
        table = Table([2000] * 3)
        result = table.play_hand(agents)
        self.assertEqual(result.returned, (180, 0, 0))
        self.assertEqual(result.awards, (50, 0, 0))
        self.assertEqual(result.ending_stacks, (2030, 1990, 1980))
        self.assertEqual(result.board, ())

    def test_all_players_tie_and_every_chip_is_conserved(self):
        table = Table([100, 200, 300, 400], evaluator=lambda cards: 0)
        result = table.play_hand([shove] * 4)
        self.assertEqual(result.ending_stacks, (100, 200, 300, 400))
        self.assertEqual(result.returned[-1], 100)


class SafetyAndProgressionTests(unittest.TestCase):
    def test_agent_reset_occurs_once_per_unique_object_per_table(self):
        class CounterAgent:
            def __init__(self):
                self.resets = 0
                self.decisions = 0
            def reset(self):
                self.resets += 1
                self.decisions = 0
            def decide(self, observation):
                self.decisions += 1
                return passive(observation)
        shared = CounterAgent()
        table = Table([500] * 3, evaluator=lambda cards: 0)
        table.play_hand([shared] * 3)
        first_hand_decisions = shared.decisions
        self.assertEqual(shared.resets, 1)
        self.assertGreater(first_hand_decisions, 3)
        table.play_hand([shared] * 3)
        self.assertEqual(shared.resets, 1)
        self.assertEqual(shared.decisions, first_hand_decisions * 2)
        Table([500] * 3, evaluator=lambda cards: 0).play_hand([shared] * 3)
        self.assertEqual(shared.resets, 2)
        self.assertEqual(shared.decisions, first_hand_decisions)

    def test_reused_actual_agent_enters_each_new_table_with_fresh_state(self):
        from agent import Agent
        class InspectAgent(Agent):
            def __init__(self):
                self.entries = []
                super().__init__(seed=14419, config={"native": False})
            def decide(self, observation):
                self.entries.append((self._hand, self.diagnostics["decisions"],
                                     tuple(tuple(row) for row in self._stats), self._rng))
                return super().decide(observation)
        agent = InspectAgent()
        first_table = Table([500] * 2, seed=419)
        first_table.play_hand([agent, passive])
        first_entry = agent.entries[0]
        self.assertIsNone(first_entry[0])
        self.assertEqual(first_entry[1], 0)
        # Deliberately leave hand_id=0 so Agent's lower-hand-ID convenience
        # reset cannot conceal a missing simulator table-boundary reset.
        agent._hand = 0
        agent._stats[1][:] = [100, 90, 120, 80]
        agent._rng = 123
        index = len(agent.entries)
        second_table = Table([500] * 2, seed=419)
        second_table.play_hand([agent, passive])
        self.assertEqual(agent.entries[index], first_entry)

    def test_private_observations_are_serializable_snapshots(self):
        agents = [Script() for _ in range(8)]
        table = Table(evaluator=lambda cards: cards[0])
        result = table.play_hand(agents, deck=range(52))
        for seat, agent in enumerate(agents):
            observation = agent.seen[0]
            self.assertEqual(observation["hole_cards"], result.hole_cards[seat])
            self.assertTrue(all("hole_cards" not in p for p in observation["players"]))
            self.assertNotIn("deck", observation)
            self.assertEqual(observation["board"], ())
            self.assertEqual(observation["pot"], sum(p["invested"] for p in observation["players"]))
            json.dumps(observation)
        self.assertEqual(len(set(card for hand in result.hole_cards for card in hand) | set(result.board)), 21)

    def test_agent_cannot_modify_other_observations_or_engine_state(self):
        def mutate(observation):
            observation["players"][1]["stack"] = 0
            observation["history"][0]["paid"] = 999999
            return passive(observation)
        agents = [mutate, Script(), Script()]
        table = Table([500] * 3, evaluator=lambda cards: 0)
        result = table.play_hand(agents)
        self.assertEqual(result.ending_stacks, (500, 500, 500))
        self.assertNotEqual(agents[1].seen[0]["players"][1]["stack"], 0)
        self.assertNotEqual(agents[1].seen[0]["history"][0]["paid"], 999999)

    def test_invalid_action_falls_back_to_fold_when_facing_bet(self):
        table = Table([200] * 2, strict=False)
        result = table.play_hand([lambda o: {"action": "check"}, passive])
        self.assertEqual(result.errors[0]["kind"], "illegal_action")
        self.assertEqual(table.invalid_actions, 1)
        self.assertEqual(result.ending_stacks, (190, 210))

    def test_exception_falls_back_and_is_counted(self):
        def broken(observation):
            raise RuntimeError("deliberate test error")
        table = Table([200] * 2, strict=False)
        result = table.play_hand([broken, passive])
        self.assertEqual(table.exceptions, 1)
        self.assertEqual(result.errors[0]["kind"], "exception")

    def test_timeout_checks_when_available(self):
        def slow_on_flop(observation):
            if observation["street"] == "flop":
                time.sleep(0.003)
            return passive(observation)
        table = Table([200] * 2, strict=False, decision_timeout_ms=1, evaluator=lambda cards: 0)
        result = table.play_hand([slow_on_flop, passive])
        self.assertEqual(table.timeouts, 1)
        self.assertEqual(result.errors[0]["kind"], "timeout")
        self.assertEqual(result.ending_stacks, (200, 200))

    def test_timeout_folds_when_facing_bet(self):
        def slow(observation):
            time.sleep(0.003)
            return passive(observation)
        table = Table([200] * 2, decision_timeout_ms=1)
        result = table.play_hand([slow, passive])
        self.assertEqual(table.timeouts, 1)
        self.assertEqual(result.ending_stacks, (190, 210))

    def test_bad_deck_is_rejected(self):
        with self.assertRaises(ValueError):
            Table([200] * 2).play_hand([passive] * 2, deck=[0] * 52)

    def test_table_enforces_configured_hand_cap(self):
        table = Table([200] * 2, max_hands=1, evaluator=lambda cards: 0)
        table.play_hand([passive] * 2)
        with self.assertRaisesRegex(ValueError, "hand cap"):
            table.play_hand([passive] * 2)

    def test_dead_small_blind_and_big_blind_progression(self):
        table = Table([500] * 5, evaluator=lambda cards: 0)
        first = table.play_hand([passive] * 5)
        self.assertEqual((first.button, first.small_blind_seat, first.big_blind_seat), (0, 1, 2))
        # Simulate an independently verified elimination between hands while
        # conserving chips. This fixture isolates blind progression from cards.
        table.players[3].stack += table.players[2].stack
        table.players[2].stack = 0
        second = table.play_hand([passive] * 5)
        self.assertEqual((second.button, second.small_blind_seat, second.big_blind_seat), (1, None, 3))
        third = table.play_hand([passive] * 5)
        self.assertEqual((third.small_blind_seat, third.big_blind_seat), (3, 4))

    def test_heads_up_transition_avoids_repeat_big_blind(self):
        table = Table([500] * 3, evaluator=lambda cards: 0)
        first = table.play_hand([passive] * 3)
        self.assertEqual(first.big_blind_seat, 2)
        table.players[0].stack += table.players[1].stack
        table.players[1].stack = 0
        second = table.play_hand([passive] * 3)
        self.assertEqual((second.button, second.small_blind_seat, second.big_blind_seat), (2, 2, 0))
        third = table.play_hand([passive] * 3)
        self.assertEqual(third.big_blind_seat, 2)

    def test_seed_reproduces_cards_stacks_and_actions(self):
        one = Table([500] * 8, seed=419, evaluator=lambda cards: cards[0]).play_hand([passive] * 8)
        two = Table([500] * 8, seed=419, evaluator=lambda cards: cards[0]).play_hand([passive] * 8)
        self.assertEqual(one, two)

    def test_random_legal_sequences_conserve_chips_and_pots(self):
        rng = random.Random(630145)
        observed = 0
        def random_agent(observation):
            nonlocal observed
            observed += 1
            hero = observation["players"][observation["hero"]]
            self.assertEqual(observation["pot"], sum(p["invested"] for p in observation["players"]))
            self.assertEqual(observation["to_call"], max(0, observation["current_bet"] - hero["committed"]))
            if observation["can_raise"] and rng.random() < 0.25:
                minimum = min(observation["min_raise_to"], observation["max_raise_to"])
                return {"action": "raise", "amount": rng.randint(minimum, observation["max_raise_to"])}
            if observation["to_call"] and rng.random() < 0.15:
                return {"action": "fold"}
            return passive(observation)
        for n in range(2, 9):
            for seed in range(8):
                table = Table([100] * n, seed=seed, small_blind=1, big_blind=2, evaluator=lambda cards: cards[0])
                for _ in range(40):
                    if sum(stack > 0 for stack in table.stacks) == 1:
                        break
                    result = table.play_hand([random_agent] * n)
                    self.assertEqual(sum(result.ending_stacks), n * 100)
                    self.assertTrue(all(stack >= 0 for stack in result.ending_stacks))
                    self.assertEqual(sum(result.awards), sum(pot.amount for pot in result.pots))
                self.assertEqual(table.invalid_actions, 0)
        self.assertGreater(observed, 1000)


class TournamentTests(unittest.TestCase):
    def test_reusing_agents_across_tournaments_resets_table_state(self):
        class CounterAgent:
            def __init__(self):
                self.resets = 0
                self.decisions = 0
            def reset(self):
                self.resets += 1
                self.decisions = 0
            def decide(self, observation):
                self.decisions += 1
                return passive(observation)
        agents = [CounterAgent(), CounterAgent()]
        options = dict(payouts=(1, 0), blind_schedule=((10, 20),), max_hands=1,
                       seed=140419, evaluator=lambda cards: 0)
        first = run_tournament(agents, **options)
        counts = tuple(agent.decisions for agent in agents)
        second = run_tournament(agents, **options)
        self.assertEqual(tuple(agent.resets for agent in agents), (2, 2))
        self.assertEqual(tuple(agent.decisions for agent in agents), counts)
        self.assertEqual(second.stacks, first.stacks)
        self.assertEqual(second.points, first.points)

    def test_hand_cap_allocates_remaining_points_by_icm(self):
        result = run_tournament(
            [passive] * 3, payouts=[0.6, 0.3, 0.1], blind_schedule=[(10, 20)],
            max_hands=1, evaluator=lambda cards: 0,
        )
        self.assertEqual(result.hands, 1)
        for value in result.points:
            self.assertAlmostEqual(value, 1 / 3)
        self.assertAlmostEqual(sum(result.points), 1)
        self.assertAlmostEqual(sum(result.finishes), 6)

    def test_simultaneous_equal_start_eliminations_split_positions(self):
        result = run_tournament(
            [shove] * 3, payouts=[0.6, 0.3, 0.1], blind_schedule=[(10, 20)],
            stacks=[100] * 3, evaluator=lambda cards: cards[0], seed=1,
        )
        # Exact winner depends on seeded cards; both equal-start losers tie.
        winner = result.stacks.index(300)
        self.assertAlmostEqual(result.points[winner], 0.6)
        for seat in range(3):
            if seat != winner:
                self.assertAlmostEqual(result.points[seat], 0.2)
                self.assertEqual(result.finishes[seat], 2.5)

    def test_simultaneous_unequal_stack_eliminations_rank_start_stacks(self):
        # Card-independent rank uses fixed hole-card markers from capture, so
        # force seat0's rank through an evaluator recognizing its deal order.
        marker = {}
        def recorder(observation):
            marker[observation["hole_cards"][0]] = observation["hero"]
            return shove(observation)
        result = run_tournament(
            [recorder] * 3, payouts=[0.6, 0.3, 0.1], blind_schedule=[(10, 20)],
            stacks=[300, 100, 200], evaluator=lambda cards: 100 if marker[cards[0]] == 0 else 0,
        )
        self.assertEqual(result.points, (0.6, 0.1, 0.3))
        self.assertEqual(result.finishes, (1.0, 3.0, 2.0))

    def test_explicit_payouts_and_blinds_required(self):
        with self.assertRaises(TypeError):
            run_tournament([passive] * 2)
        with self.assertRaises(ValueError):
            run_tournament([passive] * 2, payouts=[0, 1], blind_schedule=[(10, 20)])
        with self.assertRaises(ValueError):
            run_tournament([passive] * 2, payouts=[1, 0], blind_schedule=[])


if __name__ == "__main__":
    unittest.main()
