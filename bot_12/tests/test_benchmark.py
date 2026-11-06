from collections import Counter
import unittest
from benchmarks.evaluate import balanced_hero_seat


class BenchmarkDesignTests(unittest.TestCase):
    def test_all_seats_and_dealer_offsets_are_balanced(self):
        pairs=Counter()
        for index in range(64):
            button=(1000000+index)%8
            seat=balanced_hero_seat(index)
            pairs[(seat,(seat-button)%8)]+=1
        self.assertEqual(len(pairs),64)
        self.assertTrue(all(count==1 for count in pairs.values()))


if __name__=='__main__':
    unittest.main()
