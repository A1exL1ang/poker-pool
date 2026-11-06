import unittest
from unittest.mock import patch

from agent import Agent
from sim.engine import Hand, legal_action


class OfflineTests(unittest.TestCase):
    def test_live_action_performs_no_io_or_process_calls(self):
        bot=Agent(seed=18)
        hand=Hand([1000]*8,0)
        samples=[hand.observation()]
        while not hand.done:
            obs=hand.observation()
            if not samples or samples[-1]['street'] != obs['street']:
                samples.append(obs)
            hand.apply({'action':'call' if obs['to_call'] else 'check'})
        def forbidden(*args,**kwargs):
            raise AssertionError('live action attempted I/O')
        with patch('builtins.open',forbidden), patch('socket.socket',forbidden), patch('subprocess.Popen',forbidden):
            for obs in samples:
                legal_action(obs,bot.act(obs))
        self.assertEqual(bot.errors,0)
        self.assertEqual(len(samples),4)

    def test_table_reset_discards_statistics_and_rng_state(self):
        a,b=Agent(seed=18),Agent(seed=18)
        obs=Hand([1000]*8,0).observation()
        a.stats[4]=[100,80,60,40,30,20,10]
        a.rng.random()
        a.reset()
        self.assertEqual(a.stats,{})
        self.assertEqual(a.act(obs),b.act(obs))
        self.assertEqual(a.rng.random(),b.rng.random())


if __name__=='__main__':unittest.main()
