import unittest
from similarity_source_v2 import POLICY_FILES, function_allowed


class AuditedPolicySelectionTests(unittest.TestCase):
    def test_explicit_policy_sources_cover_exactly_twenty(self):
        self.assertEqual(set(POLICY_FILES),set(range(1,21)))
        self.assertEqual(POLICY_FILES[18],('agent.py',))
        self.assertEqual(POLICY_FILES[14],('pokerbot/strategy.py',))
        self.assertEqual(POLICY_FILES[19],('agent.py',))
        for paths in POLICY_FILES.values():
            for path in paths:
                self.assertNotIn(path,('policy/icm_fast.py','pokerbot/game.py','sim.py','model.py'))

    def test_embedded_mechanical_code_excluded(self):
        for name in ('from_dict','normalize','observation','_icm_hero','_pot_layers','_random','_raise','reset'):
            self.assertFalse(function_allowed(9,name),name)
        self.assertFalse(function_allowed(19,'predict_from_features'))
        self.assertFalse(function_allowed(9,'equity_vs_range'))

    def test_actual_betting_and_range_logic_retained(self):
        for number in range(1,21):
            self.assertTrue(function_allowed(number,'_preflop'))
            self.assertTrue(function_allowed(number,'_postflop'))
            self.assertTrue(function_allowed(number,'_ranges'))
            self.assertTrue(function_allowed(number,'_observe'))


if __name__=='__main__':unittest.main()
