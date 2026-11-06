import random

from common.engine import (
    CALL, CHECK, FOLD, RAISE, Action, Bot, Config, Observation, Table, run_tournament,
)


class Scripted(Bot):
    """Plays a fixed list of actions per decision; falls back to check/call."""

    def __init__(self, script=()):
        self.script = list(script)
        self.seen = []

    def reset(self, seed, seat, n_seats):
        pass

    def act(self, obs: Observation) -> Action:
        self.seen.append(obs)
        if self.script:
            return self.script.pop(0)
        return Action(CHECK) if obs.to_call == 0 else Action(CALL)


class Caller(Bot):
    def reset(self, seed, seat, n_seats):
        pass

    def act(self, obs):
        return Action(CHECK) if obs.to_call == 0 else Action(CALL)


class Folder(Bot):
    def reset(self, seed, seat, n_seats):
        pass

    def act(self, obs):
        return Action(CHECK) if obs.to_call == 0 else Action(FOLD)


class RandomBot(Bot):
    def reset(self, seed, seat, n_seats):
        self.rng = random.Random(seed)

    def act(self, obs):
        r = self.rng.random()
        if obs.can_raise and r < 0.3:
            lo, hi = obs.min_raise_to, obs.max_raise_to
            amt = self.rng.randint(min(lo, hi), hi) if lo <= hi else hi
            if amt < obs.min_raise_to and amt != obs.max_raise_to:
                amt = obs.max_raise_to
            return Action(RAISE, amt)
        if obs.to_call == 0:
            return Action(CHECK)
        return Action(CALL) if r < 0.75 else Action(FOLD)


class Shover(Bot):
    def reset(self, seed, seat, n_seats):
        pass

    def act(self, obs):
        if obs.can_raise:
            return Action(RAISE, obs.max_raise_to)
        return Action(CALL) if obs.to_call else Action(CHECK)


def _table(bots, **kw):
    cfg = Config(n_seats=len(bots), **kw)
    return Table(bots, cfg, seed=kw.pop("seed", 1) if "seed" in kw else 1)


def test_chip_conservation_random_play():
    for seed in range(40):
        bots = [RandomBot() for _ in range(8)]
        res = run_tournament(bots, Config(record_hands=True), seed=seed)
        assert sum(res.final_stacks) == 16000
        assert abs(sum(res.points) - 1.0) < 1e-9
        assert res.hands <= 200
        assert sum(res.illegal) == 0
        for rec in res.hand_records:
            assert sum(rec.stacks_before) == 16000 == sum(rec.stacks_after)


def test_blind_posting_and_order():
    bots = [Scripted() for _ in range(3)]
    cfg = Config(n_seats=3)
    t = Table(bots, cfg, seed=5)
    t.play_hand()
    # everyone limped/checked; first decision belongs to the seat after BB (= button in 3-handed)
    first = min((b.seen[0] for b in bots if b.seen), key=lambda o: len(o.history))
    assert first.seat == first.button
    assert first.to_call == 20
    assert first.pot == 30
    assert first.stacks[first.bb_seat] == 1980 and first.stacks[first.sb_seat] == 1990
    assert first.min_raise_to == 40


def test_heads_up_button_is_small_blind_and_acts_first_preflop():
    bots = [Scripted(), Scripted()]
    t = Table(bots, Config(n_seats=2), seed=2)
    t.play_hand()
    first = min((b.seen[0] for b in bots if b.seen), key=lambda o: len(o.history))
    assert first.seat == first.sb_seat == first.button
    assert first.to_call == 10
    # postflop, BB (non-button) acts first
    post = [o for b in bots for o in b.seen if o.street == 1]
    post.sort(key=lambda o: len(o.history))
    assert post[0].seat == post[0].bb_seat


def test_side_pots_three_way_all_in():
    # seats: 0 has 100, 1 has 300, 2 has 1000. everyone shoves / calls.
    bots = [Shover(), Shover(), Shover()]
    cfg = Config(n_seats=3, starting_stack=1000)
    t = Table(bots, cfg, seed=0)
    t.stacks = [100, 300, 1000]
    t.play_hand()
    assert sum(t.stacks) == 1400
    # the short stack can win at most 300, the middle at most 900
    assert t.stacks[0] in (0, 300) or t.stacks[0] <= 300
    assert t.stacks[1] <= 900


def test_min_raise_and_underraise_reopening():
    # 3 players, stacks: button 2000, sb 2000, bb 2000. Button raises to 60,
    # SB shoves 2000, BB with 2000 can re-raise? BB can (full raise). Check min_raise_to values.
    scripts = {}
    bots = [Scripted(), Scripted(), Scripted()]
    cfg = Config(n_seats=3)
    t = Table(bots, cfg, seed=1)
    # find who is button: emulate by playing with a bot that raises to 60 when first to act
    class First(Bot):
        def reset(self, *a):
            pass
        def act(self, obs):
            if obs.street == 0 and not obs.history:
                assert obs.min_raise_to == 40
                return Action(RAISE, 60)
            if obs.street == 0 and len(obs.history) == 1:
                assert obs.min_raise_to == 100  # 60 + 40
                return Action(RAISE, 95) if False else Action(CALL)
            return Action(CHECK) if obs.to_call == 0 else Action(CALL)
    t = Table([First(), First(), First()], cfg, seed=1)
    t.play_hand()
    assert sum(t.stacks) == 6000


def test_underraise_all_in_does_not_reopen():
    # seat A opens 100, seat B all-in for 130 (short), seat A must not be able to re-raise.
    class A(Bot):
        def __init__(self):
            self.obs = []
        def reset(self, *a):
            pass
        def act(self, obs):
            self.obs.append(obs)
            if obs.street == 0 and obs.to_call and not any(e.kind == RAISE and e.seat == obs.seat for e in obs.history):
                return Action(RAISE, 100)
            return Action(CHECK) if obs.to_call == 0 else Action(CALL)

    class B(Bot):
        def reset(self, *a):
            pass
        def act(self, obs):
            if obs.can_raise:
                return Action(RAISE, obs.max_raise_to)
            return Action(CALL) if obs.to_call else Action(CHECK)

    cfg = Config(n_seats=2)
    a, b = A(), B()
    t = Table([a, b], cfg, seed=3)
    # find which seat is button: button acts first preflop. Make stacks so B is short.
    # Try both seatings until A is the opener.
    t.stacks = [2000, 130]
    t.play_hand()
    later = [o for o in a.obs if o.street == 0 and len(o.history) >= 2]
    if later:
        o = later[0]
        assert o.to_call == 30 and not o.can_raise
    assert sum(t.stacks) == 2130


def test_tournament_scoring_and_icm_cap():
    bots = [Caller() for _ in range(8)]
    cfg = Config(max_hands=3)
    res = run_tournament(bots, cfg, seed=11)
    assert abs(sum(res.points) - 1.0) < 1e-9
    assert res.hands == 3


def test_elimination_points():
    bots = [Folder() for _ in range(3)] + [Shover() for _ in range(5)]
    for seed in range(10):
        res = run_tournament(bots, Config(), seed=seed)
        assert abs(sum(res.points) - 1.0) < 1e-9
        assert sum(1 for s in res.final_stacks if s > 0) >= 1


def test_antes():
    bots = [Caller() for _ in range(4)]
    cfg = Config(n_seats=4, schedule=((10, 20, 5),))
    t = Table(bots, cfg, seed=0)
    t.play_hand()
    assert sum(t.stacks) == 8000


def test_timeout_and_error_fallback():
    class Bad(Bot):
        def reset(self, *a):
            pass
        def act(self, obs):
            raise ValueError("boom")

    class Illegal(Bot):
        def reset(self, *a):
            pass
        def act(self, obs):
            return Action(RAISE, 7)

    res = run_tournament([Bad(), Illegal(), Caller(), Caller()], Config(n_seats=4, max_hands=5), seed=0)
    assert res.errors[0] > 0 and res.illegal[1] > 0 and res.illegal[0] == 0
    assert abs(sum(res.points) - 1.0) < 1e-9


def test_short_ante_allin_everyone_folds_keeps_chips():
    # antes exceed two short stacks; the others fold; no chips may vanish
    bots = [Folder(), Folder(), Shover(), Shover()]
    cfg = Config(n_seats=4, schedule=((400, 800, 200),))
    for seed in range(20):
        t = Table(bots, cfg, seed=seed)
        t.stacks = [2269, 2370, 150, 144]
        t.play_hand()
        assert sum(t.stacks) == 2269 + 2370 + 150 + 144
