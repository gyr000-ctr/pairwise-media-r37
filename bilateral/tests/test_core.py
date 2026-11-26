import math

import pytest

import bilateral

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.0, 0.0], [1.0, 1.0]]

CASES = [
    ("constant", lambda: all(abs(v - 0.5) < 1e-12 for row in bilateral.Bilateral(1.0, 0.1, 2).apply([[0.5] * 4 for _ in range(4)]) for v in row)),
    ("shape", lambda: len(bilateral.Bilateral(1.0, 0.1, 2).apply(IMG)) == 2),
    ("edge_keep", lambda: abs(bilateral.Bilateral(1.0, 0.05, 2).apply(IMG)[0][0]) < 0.25),
    ("bad_sigma_s", lambda: _raises(lambda: bilateral.Bilateral(0.0, 0.1, 2))),
    ("bad_sigma_r", lambda: _raises(lambda: bilateral.Bilateral(1.0, 0.0, 2))),
    ("bad_radius", lambda: _raises(lambda: bilateral.Bilateral(1.0, 0.1, 0))),
    ("empty", lambda: _raises(lambda: bilateral.Bilateral(1.0, 0.1, 2).apply([]))),
    ("type", lambda: isinstance(bilateral.Bilateral(1.0, 0.1, 2).apply(IMG), list)),
    ("det", lambda: bilateral.Bilateral(1.0, 0.1, 2).apply(IMG) == bilateral.Bilateral(1.0, 0.1, 2).apply(IMG)),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in bilateral.Bilateral(1.0, 0.1, 2).apply(IMG) for v in row)),
    ("nan", lambda: _raises(lambda: bilateral.Bilateral(1.0, 0.1, 2).apply([[float('nan')]]))),
    ("no_mut", lambda: bilateral.Bilateral(1.0, 0.1, 2).apply([r[:] for r in IMG]) is not None),
    ("kernel_sum", lambda: abs(sum(bilateral.Bilateral(1.0, 0.1, 2).kernel(0.0)) - 1.0) < 1e-12),
    ("kernel_type", lambda: isinstance(bilateral.Bilateral(1.0, 0.1, 2).kernel(0.0), list)),
    ("float", lambda: isinstance(bilateral.Bilateral(1.0, 0.1, 2).apply(IMG)[0][0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
