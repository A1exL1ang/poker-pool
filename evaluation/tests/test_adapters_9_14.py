"""Run remotely: PYTHONPATH=evaluation python -m unittest discover -s evaluation/tests -p test_adapters_9_14.py -v."""
import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters_9_14 import Adapter, convert, public_history
from common.engine import Action, Config, Event, Observation, run_tournament
import isolated_loader


def observation(**changes):
    values = dict(seat=3, hole=(48, 49), board=(), n_seats=8,
                  stacks=(2000, 1990, 1980, 2000, 2000, 2000, 2000, 2000),
                  street_bets=(0, 10, 20, 0, 0, 0, 0, 0),
                  contributions=(0, 10, 20, 0, 0, 0, 0, 0),
                  folded=(False,) * 8, all_in=(False,) * 8, active=(True,) * 8,
                  starting_stacks=(2000,) * 8, button=0, sb_seat=1, bb_seat=2,
                  sb=10, bb=20, ante=0, pot=30, to_call=20,
                  min_raise_to=40, max_raise_to=2000, can_raise=True,
                  street=0, hand_number=0, hands_left=200, history=(),
                  payouts=(.5, .3, .2, 0, 0, 0, 0, 0))
    values.update(changes)
    return Observation(**values)


class ConversionTests(unittest.TestCase):
    def test_contracts_preserve_chips_and_cards(self):
        o = observation()
        for n in range(9, 15):
            d = convert(n, o)
            self.assertEqual(d['pot'], 30)
            self.assertEqual(d['to_call'], 20)
            self.assertEqual(d['hole'], (48, 49))
            self.assertEqual(d['board'], ())
            self.assertNotIn('seed', d)
            self.assertNotIn('table_id', d)
        self.assertEqual(convert(9, o)['committed'][1], 10)
        self.assertEqual(convert(11, o)['players'][2]['street_bet'], 20)
        self.assertEqual(convert(12, o)['street_contributions'][2], 20)
        self.assertEqual(convert(14, o)['players'][1]['invested'], 10)

    def test_history_call_increment_and_raise_total(self):
        o = observation(history=(Event(3, 0, 'raise', 60, 60),
                                 Event(1, 0, 'call', 60, 50),
                                 Event(2, 0, 'fold', 0, 0)))
        h = public_history(o)
        self.assertEqual(h[0]['pot_before'], 30)
        self.assertEqual(h[0]['to_call_before'], 20)
        self.assertEqual(h[1]['amount'], 50)
        self.assertEqual(h[1]['pot_before'], 90)
        self.assertEqual(h[1]['stack_before'], 1990)
        self.assertEqual(h[1]['current_bet_before'], 60)
        self.assertEqual(public_history(o, total_calls=True)[1]['amount'], 60)
        self.assertEqual(convert(13, o)['history'][1]['amount_paid'], 50)
        self.assertEqual(convert(14, o)['history'][0]['action'], 'post_sb')
        self.assertEqual(convert(10, o)['history'][0]['action'], 'small_blind')

    def test_short_stack_and_reopening_contract(self):
        o = observation(stacks=(2000, 1990, 1980, 5, 0, 0, 0, 0),
                        active=(True,) * 4 + (False,) * 4,
                        folded=(False,) * 4 + (True,) * 4,
                        can_raise=False, max_raise_to=5)
        for n in range(9, 14):
            d = convert(n, o)
            self.assertEqual(d['to_call'], 5)
            self.assertNotIn('raise', d['legal_actions'])
        self.assertEqual(convert(14, o)['to_call'], 20)
        self.assertEqual(len(convert(10, o)['payouts']), 4)
        self.assertEqual(len(convert(11, o)['players']), 8)
        self.assertFalse(convert(12, o)['in_hand'][7])

    def test_street_transition_history(self):
        o = observation(history=(Event(3, 0, 'call', 20, 20),
                                 Event(1, 1, 'raise', 40, 40),
                                 Event(2, 1, 'call', 40, 40)))
        h = public_history(o)
        self.assertEqual(h[1]['current_bet_before'], 0)
        self.assertEqual(h[1]['pot_before'], 50)
        self.assertEqual(h[2]['to_call_before'], 40)
        self.assertEqual(h[2]['street'], 'flop')


class ImportIsolationTests(unittest.TestCase):
    def test_flat_package_and_lazy_imports_are_private(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = isolated_loader.ROOT
            try:
                isolated_loader.ROOT = Path(temporary)
                for n in (901, 902):
                    p = Path(temporary) / f'bot_{n}'
                    p.mkdir()
                    (p / 'shared.py').write_text(f'value={n}\n')
                    (p / 'pkg').mkdir()
                    (p / 'pkg/__init__.py').write_text('')
                    (p / 'pkg/sub.py').write_text('from shared import value\n')
                    (p / 'agent.py').write_text(
                        'import pkg.sub\nfrom shared import value\n'
                        'def lazy():\n from shared import value\n return value\n')
                a = isolated_loader.load_bot_module(901)
                b = isolated_loader.load_bot_module(902)
                self.assertEqual((a.value, a.lazy(), a.pkg.sub.value), (901,) * 3)
                self.assertEqual((b.value, b.lazy(), b.pkg.sub.value), (902,) * 3)
                self.assertNotIn('shared', sys.modules)
                self.assertNotIn('pkg', sys.modules)
                self.assertIn('bot_901', a.__file__)
            finally:
                isolated_loader.ROOT = saved
                for name in list(sys.modules):
                    if name.startswith(('isolated_submission_901', 'isolated_submission_902')):
                        del sys.modules[name]
                isolated_loader._ROOTS.pop('isolated_submission_901', None)
                isolated_loader._ROOTS.pop('isolated_submission_902', None)


class RuntimeIntegrationTests(unittest.TestCase):
    def test_six_bots_coexist_and_play_without_errors(self):
        # Intentional remote integration check, not a strength measurement.
        bots = [Adapter(n) for n in (9, 10, 11, 12, 13, 14, 9, 14)]
        result = run_tournament(bots, Config(max_hands=12, strict_timing=False), 131901)
        self.assertEqual(sum(result.final_stacks), 16000)
        self.assertEqual(sum(result.errors), 0)
        self.assertEqual(sum(result.illegal), 0)
        self.assertEqual(sum(b.internal_errors for b in bots), 0)
        self.assertTrue(all(b.calls > 0 for b in bots))
        # Imports and lazy imports must not pollute names owned by another bot.
        for name in ('cards', 'equity', 'model', 'pokerbot', 'bot13'):
            self.assertNotIn(name, sys.modules)

    def test_each_street_and_busted_seat(self):
        for n in range(9, 15):
            b = Adapter(n)
            b.reset(773 + n, 3, 8)
            for street, board in enumerate(((), (0, 5, 10), (0, 5, 10, 15), (0, 5, 10, 15, 20))):
                o = observation(street=street, board=board, hand_number=street,
                                active=(True,) * 4 + (False,) * 4,
                                folded=(False,) * 4 + (True,) * 4,
                                stacks=(2000, 1990, 1980, 2000, 0, 0, 0, 0))
                action = b.act(o)
                self.assertIn(action.kind, ('fold', 'check', 'call', 'raise'))
            self.assertEqual(b.internal_errors, 0, n)


if __name__ == '__main__':
    unittest.main()
