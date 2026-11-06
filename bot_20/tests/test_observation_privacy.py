"""Public table tags must not reveal the deterministic deal seed."""
import unittest

from simulator.baselines import make_baseline
from simulator.engine import play_tournament


class ObservationPrivacyTests(unittest.TestCase):
    def test_table_identifier_is_independent_of_deal_seed(self):
        identifiers = []
        for _ in range(2):
            observations = []
            play_tournament([make_baseline("station", 1), make_baseline("station", 2)],
                            seed=842319, max_hands=1, blind_levels=[(10, 20)],
                            payouts=[1, 0], on_observation=observations.append)
            current = {obs["table_id"] for obs in observations}
            self.assertEqual(len(current), 1)
            identifier = current.pop()
            self.assertNotEqual(identifier, 842319)
            self.assertNotEqual(identifier, "842319")
            identifiers.append(identifier)
        self.assertNotEqual(*identifiers)


if __name__ == "__main__":
    unittest.main()
