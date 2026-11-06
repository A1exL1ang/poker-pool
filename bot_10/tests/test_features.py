"""Feature invariants and optional native-teacher parity checks.

Set VERIFY_CPP=1 to compile the original C++ teacher and compare 3,000 random
feature vectors plus evaluator ranks. This requires g++ only for verification.
"""
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from cards import evaluate, parse_card
from features import features

ROOT = Path(__file__).resolve().parents[1]


class FeatureTests(unittest.TestCase):
    def test_shape_and_made_hand_encoding(self):
        examples = [
            ("As 2s", "3s 4s 5s", 8),
            ("As Ad", "Ac Ah Ks", 7),
            ("As Ad", "Ac Ks Kd", 6),
            ("As Ks", "Qs 9s 8s", 5),
            ("As 2d", "3c 4h 5s", 4),
            ("As Ad", "Ac Kh Qs", 3),
            ("As Ad", "Ks Kh Qs", 2),
            ("As Ad", "Ks Qh Js", 1),
            ("As Kd", "Js 9h 7s", 0),
        ]
        for h, b, category in examples:
            hole = [parse_card(x) for x in h.split()]
            board = [parse_card(x) for x in b.split()]
            value = features(hole, board)
            self.assertEqual(len(value), 56)
            self.assertEqual(value[36], category)
            self.assertEqual(value[37:42], [(evaluate(hole + board) >> shift) & 15 for shift in (16, 12, 8, 4, 0)])

    def test_board_order_and_suit_invariance(self):
        rng = random.Random(813)
        for _ in range(1000):
            deal = rng.sample(range(52), 7)
            hole, board = deal[:2], deal[2:2 + rng.randrange(3, 6)]
            mapping = rng.sample(range(4), 4)
            permute = lambda cards: [c // 4 * 4 + mapping[c % 4] for c in cards]
            self.assertEqual(features(hole, board), features(permute(hole), permute(board[::-1])))

    @unittest.skipUnless(os.environ.get("VERIFY_CPP") == "1" and shutil.which("g++"), "set VERIFY_CPP=1 and install g++ for native teacher parity")
    def test_cpp_teacher_parity(self):
        source = ROOT / "tools" / "generate_equity.cpp"
        with tempfile.TemporaryDirectory(prefix="feature-check-", dir=ROOT / ".control") as temp:
            temp = Path(temp)
            harness = temp / "harness.cpp"
            exe = temp / ("harness.exe" if os.name == "nt" else "harness")
            harness.write_text('#define main original_teacher_main\n#include "' + source.as_posix() + '"\n#undef main\n'
                               'int main(){int n,a,b,c;while(cin>>n>>a>>b){vector<int> h={a,b}, board;'
                               'for(int i=0;i<n;i++){cin>>c;board.push_back(c);}auto f=features(h,board);'
                               'auto all=h;all.insert(all.end(),board.begin(),board.end());'
                               'cout<<evaluate(all);for(int x:f)cout<<","<<x;cout<<"\\n";}return 0;}\n', encoding="utf-8")
            subprocess.run(["g++", "-std=c++17", "-O2", str(harness), "-o", str(exe)], check=True, capture_output=True)
            rng = random.Random(2083901)
            states = []
            for row in range(3000):
                deal = rng.sample(range(52), 7)
                states.append((deal[:2], deal[2:5 + row % 3]))
            data = "".join(" ".join(map(str, [len(b)] + h + b)) + "\n" for h, b in states)
            out = subprocess.run([str(exe)], input=data, text=True, capture_output=True, check=True)
            lines = out.stdout.splitlines()
            self.assertEqual(len(lines), len(states))
            for line, (h, b) in zip(lines, states):
                values = list(map(int, line.split(",")))
                self.assertEqual(values[0], evaluate(h + b), (h, b))
                self.assertEqual(values[1:], features(h, b), (h, b))


if __name__ == "__main__":
    unittest.main()
