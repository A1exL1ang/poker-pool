"""Run on an allocated compute node: python -m unittest test_adapters_15_20 -v."""
import dataclasses
import unittest
from types import SimpleNamespace
from common.engine import Observation, Event, Config, run_tournament
from adapters_15_20 import Adapter, observation_for, public_events


def example():
    return Observation(seat=0, hole=(48,49), board=(), n_seats=8,
        stacks=(2000,1990,1980,2000,2000,2000,2000,2000),
        street_bets=(0,10,20,0,0,0,0,0), contributions=(0,10,20,0,0,0,0,0),
        folded=(False,)*8, all_in=(False,)*8, active=(True,)*8,
        starting_stacks=(2000,)*8, button=0, sb_seat=1, bb_seat=2,
        sb=10,bb=20,ante=0,pot=30,to_call=20,min_raise_to=40,max_raise_to=2000,
        can_raise=True,street=0,hand_number=0,hands_left=200,history=(),
        payouts=(.5,.3,.2,0,0,0,0,0))


class TranslationTests(unittest.TestCase):
    def test_call_caps_and_street_names(self):
        o=dataclasses.replace(example(),stacks=(5,1990,1980,2000,2000,2000,2000,2000),max_raise_to=5,can_raise=False)
        for n in range(15,21):
            d=observation_for(n,o,'opaque')
            self.assertEqual(d['to_call'],5 if n in (17,18) else 20)
            self.assertEqual(d['street'],'preflop' if n in (18,19) else 0)
            self.assertEqual(d['payouts'],list(o.payouts))
            self.assertEqual(d['table_id'],'opaque')

    def test_history_chips_and_street_reset(self):
        o=dataclasses.replace(example(),history=(Event(3,0,'raise',60,60),Event(1,0,'call',60,50),Event(2,0,'fold',0,0),Event(1,1,'raise',80,80)))
        e=public_events(o)
        self.assertEqual([x['pot_before'] for x in e],[30,90,140,140])
        self.assertEqual([x['to_call_before'] for x in e],[20,50,40,0])
        self.assertEqual(e[1]['stack_before'],1990)
        self.assertEqual(e[3]['stack_before'],1940)
        self.assertEqual(observation_for(15,o,'x')['history'][1]['amount'],50)
        self.assertEqual(observation_for(16,o,'x')['history'][1]['to'],60)
        self.assertEqual(observation_for(17,o,'x')['history'][1]['amount'],60)
        self.assertEqual(observation_for(18,o,'x')['history'][1]['raise_to'],60)
        self.assertEqual(observation_for(19,o,'x')['history'][1]['amount'],50)
        self.assertEqual(observation_for(20,o,'x')['history'][1]['amount'],50)

    def test_eliminated_seat_and_dead_money_retained(self):
        o=dataclasses.replace(example(),active=(True,True,True,False,True,True,True,True),
            stacks=(2000,1990,1980,0,2000,2000,2000,2000),starting_stacks=(2000,2000,2000,0,2000,2000,2000,2000))
        for n in range(15,21):
            d=observation_for(n,o,'x')
            if 'players' in d:
                self.assertEqual(len(d['players']),8)
                self.assertTrue(d['players'][3]['folded'])
            elif n==15:self.assertFalse(d['in_hand'][3])
            else:self.assertTrue(d['folded'][3])
            self.assertEqual(d['pot'],30)
            self.assertEqual(len(d['payouts']),8)

    def test_position_headsup_and_full_table(self):
        self.assertEqual(observation_for(15,example(),'x')['position'],2)
        o=dataclasses.replace(example(),active=(True,False,True,False,False,False,False,False),
            folded=(False,True,False,True,True,True,True,True),sb_seat=0,bb_seat=2)
        self.assertEqual(observation_for(15,o,'x')['position'],1)

    def test_observation_contains_no_seed_or_other_cards(self):
        o=example()
        for n in range(15,21):
            d=observation_for(n,o,'opaque')
            self.assertNotIn('seed',d)
            self.assertNotIn('deck',d)
            self.assertNotIn('opponent_cards',d)
            self.assertEqual(d.get('hole',d.get('hole_cards')),list(o.hole))

    def test_counters_do_not_double_count_safety(self):
        a=Adapter.__new__(Adapter);a.number=16;a.table_id='x';a.calls=0
        a.bot=SimpleNamespace(errors=3,fallbacks=3,budget_fallbacks=2,
                              act=lambda o:{'action':'call'})
        a.act(example())
        self.assertEqual(a.diagnostic_counters(),dict(internal_errors=3,safety_fallbacks=3,budget_fallbacks=2))


class RuntimeIntegrationTests(unittest.TestCase):
    def test_coexisting_imports_and_all_streets(self):
        adapters=[Adapter(n) for n in range(15,21)]
        for a in adapters:a.reset(710+a.number,0,8)
        # Constructor and lazy imports remain bound to each private submission.
        self.assertEqual(len({a.module.__name__ for a in adapters}),6)
        for street,board in ((0,()),(1,(0,5,10)),(2,(0,5,10,15)),(3,(0,5,10,15,24))):
            o=example()
            if street:
                o=dataclasses.replace(o,street=street,board=board,hand_number=street,
                    street_bets=(0,)*8,to_call=0,min_raise_to=20)
            for a in adapters:
                action=a.act(o)
                self.assertIn(action.kind,('fold','check','call','raise'))
                self.assertEqual(a.internal_errors,0,f'bot {a.number} internal safety fallback')
        for a in adapters:
            old=a.table_id;a.reset(999,0,8)
            self.assertNotEqual(a.table_id,old)
            self.assertNotIn('999',a.table_id)

    def test_mixed_table_legal_actions(self):
        ids=(15,16,17,18,19,20,15,16)
        agents=[Adapter(n) for n in ids]
        result=run_tournament(agents,Config(max_hands=12,strict_timing=False),seed=802134)
        self.assertEqual(sum(result.errors),0)
        self.assertEqual(sum(result.illegal),0)
        self.assertEqual(sum(a.internal_errors for a in agents),0)
        self.assertEqual(sum(result.final_stacks),16000)
        self.assertAlmostEqual(sum(result.points),1)


if __name__=='__main__':unittest.main()
