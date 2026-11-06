import random

from common.icm import icm, icm_reference


def test_icm_matches_reference():
    rng = random.Random(3)
    for _ in range(300):
        n = rng.randint(2, 7)
        stacks = [rng.randint(0, 5000) for _ in range(n)]
        if sum(stacks) == 0:
            stacks[0] = 10
        k = rng.randint(1, n)
        payouts = [rng.random() for _ in range(k)]
        payouts.sort(reverse=True)
        a = icm(stacks, payouts)
        b = icm_reference(stacks, payouts)
        for x, y in zip(a, b):
            assert abs(x - y) < 1e-9, (stacks, payouts, a, b)


def test_icm_properties():
    ev = icm([2000] * 8, [0.5, 0.3, 0.2])
    assert all(abs(x - 0.125) < 1e-12 for x in ev)
    ev = icm([1000, 3000], [1.0])
    assert abs(ev[0] - 0.25) < 1e-12
    ev = icm([1000, 3000, 0], [0.6, 0.4])
    assert ev[2] == 0 and abs(sum(ev) - 1.0) < 1e-12
    ev = icm([1, 1, 1, 1, 1, 1, 1, 1], [8, 7, 6, 5, 4, 3, 2, 1])
    assert all(abs(x - 4.5) < 1e-9 for x in ev)


# The upstream preflop-training solver test is not copied: that solver is not
# part of the common tournament engine. Engine conservation is tested separately.
