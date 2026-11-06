"""Rule/adapter regression checks for harness v2; execute on allocated compute."""
import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from common.engine import Action, Config, Event, Table, run_tournament
from adapters import Adapter as LegacyAdapter, history
from adapters_15_20 import public_events
from test_adapters_15_20 import example


class CallingBot:
    def reset(self,*args):pass
    def act(self,o):return Action('call' if o.to_call else 'check')


class PoolRuleTests(unittest.TestCase):
    def test_short_blind_public_history_uses_callable_live_commitment(self):
        o=dataclasses.replace(example(),starting_stacks=(100,10,5,0,0,0,0,0),
            active=(True,True,True,False,False,False,False,False),
            history=(Event(0,0,'call',10,10),))
        self.assertEqual(history(o)[0]['to_call'],10)
        self.assertEqual(public_events(o)[0]['to_call_before'],10)

    def test_short_blind_history_updates_after_last_responder_folds(self):
        o=dataclasses.replace(example(),starting_stacks=(100,10,5,100,0,0,0,0),
            active=(True,True,True,True,False,False,False,False),
            history=(Event(3,0,'fold',0,0),Event(0,0,'call',10,10)))
        self.assertEqual(history(o)[0]['to_call'],20)
        self.assertEqual(history(o)[1]['to_call'],10)
        self.assertEqual(public_events(o)[1]['to_call_before'],10)

    def test_legacy_table_id_is_not_deck_seed(self):
        a=LegacyAdapter.__new__(LegacyAdapter);a.number=1
        a.module=SimpleNamespace(Agent=lambda seed:SimpleNamespace(seed=seed))
        a.reset(2026110600*1000003,0,8)
        first=a.table_id
        self.assertNotEqual(first,'2026110600')
        a.reset(2026110600*1000003,0,8)
        self.assertNotEqual(a.table_id,first)

    def test_bot1_explicit_fallback_counted_once_per_report(self):
        a=LegacyAdapter.__new__(LegacyAdapter);a.number=1;a.table_id='opaque'
        a.calls=0;a.internal_errors=0
        bot=SimpleNamespace(last_info={},last_error=None)
        def fallback(o):
            bot.last_info={'fallback':'invalid observation'}
            return {'action':'fold'}
        bot.act=fallback;a.bot=bot
        a.act(example())
        self.assertEqual(a.internal_errors,1)
        a.act(example())
        self.assertEqual(a.internal_errors,2)
        # Successful early returns can retain the previous diagnostic object.
        bot.act=lambda o:{'action':'call'}
        a.act(example())
        self.assertEqual(a.internal_errors,2)
        bot.act=fallback;bot.last_error='same failure reported through two fields'
        with self.assertRaises(RuntimeError):a.act(example())
        self.assertEqual(a.internal_errors,3)

    def test_sidepots_and_folded_dead_money(self):
        t=Table([CallingBot() for _ in range(4)],Config(n_seats=4),13)
        t.stacks=[0]*4
        t._settle([100,200,200,50],[False,False,True,False],{0:10,1:5,3:20},0,list(range(4)))
        self.assertEqual(t.stacks,[150,200,0,200])

    def test_uncalled_excess_returned_before_sidepots(self):
        t=Table([CallingBot() for _ in range(3)],Config(n_seats=3),13)
        t.stacks=[0]*3
        t._settle([100,40,40],[False,False,False],{0:5,1:10,2:10},0,list(range(3)))
        self.assertEqual(t.stacks,[60,60,60])

    def test_cumulative_short_allins_reopen_individually(self):
        observations=[]
        class Script(CallingBot):
            def act(self,o):
                observations.append(o)
                if o.street==0:
                    if not o.history:return Action('raise',40)
                    if o.seat==1:return Action('raise',50)
                    if o.seat==2:return Action('raise',60)
                return super().act(o)
        t=Table([Script() for _ in range(4)],Config(n_seats=4,max_hands=1),21)
        t.bb_seat=1;t.stacks=[200,50,60,200];t.play_hand()
        returning=[o for o in observations if o.street==0 and o.seat in (0,3) and len(o.history)>=4]
        self.assertEqual(len(returning),2)
        self.assertTrue(all(o.can_raise for o in returning))
        self.assertTrue(all(o.min_raise_to==80 for o in returning))
        self.assertEqual(sum(t.stacks),510)
        self.assertEqual(sum(t.illegal),0)

    def test_icm_cap_conserves_remaining_awards(self):
        t=Table([CallingBot() for _ in range(4)],Config(n_seats=4,max_hands=0,payouts=(.5,.3,.2,0)),21)
        t.stacks=[0,100,200,300];t.finished=[[0]]
        r=t._score()
        self.assertEqual(r.points[0],0)
        self.assertAlmostEqual(sum(r.points),1)
        self.assertGreater(r.points[3],r.points[2])
        self.assertGreater(r.points[2],r.points[1])

    def test_timing_fallback_accounted_separately(self):
        ticks=iter([0,2_000_000]*100)
        with patch('common.engine.perf_counter_ns',side_effect=lambda:next(ticks)):
            r=run_tournament([CallingBot() for _ in range(8)],Config(max_hands=1,strict_timing=True),11)
        self.assertGreater(sum(r.timeouts),0)
        self.assertEqual(sum(r.errors),0)
        self.assertEqual(sum(r.illegal),0)
        self.assertEqual(sum(r.final_stacks),16000)
        self.assertAlmostEqual(sum(r.points),1)


if __name__=='__main__':unittest.main()
