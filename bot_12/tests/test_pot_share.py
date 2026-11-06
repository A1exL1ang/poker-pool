import unittest
from agent import _multiway_equity, _range_equity


class PotShareTests(unittest.TestCase):
    def test_guaranteed_chop_shares_and_heads_up_limits(self):
        for opponents in range(1,8):
            self.assertAlmostEqual(_multiway_equity(.5,opponents,1.),1/(opponents+1))
        for q in (.0,.2,.5,.8,1.):
            self.assertAlmostEqual(_multiway_equity(q,1,min(q,1-q)),q)

    def test_two_opponents_independent_win_and_tie_events(self):
        # P(win both) + P(one tie, one win)/2 + P(tie both)/3.
        w,t=.3,.4
        self.assertAlmostEqual(_multiway_equity(w+t/2,2,t),w*w+w*t+t*t/3)

    def test_narrow_range_interpolation_uses_its_own_curve(self):
        wide=(.7,.72,.74,.77,.8,.85)
        narrow=(.4,.6)
        self.assertEqual(_range_equity(wide,.01,narrow),.4)
        self.assertAlmostEqual(_range_equity(wide,.025,narrow),.6)
        self.assertAlmostEqual(_range_equity(wide,.0375,narrow),.65)

    def test_out_of_range_approximation_cannot_create_complex_equity(self):
        self.assertEqual(_multiway_equity(-.01,2),0.)
        self.assertEqual(_multiway_equity(1.01,3),1.)


if __name__=='__main__':
    unittest.main()
