import math

import pytest

import colormatrix

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

M = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
C = (0.4, 0.5, 0.6)

CASES = [
    ("identity", lambda: all(abs(a - b) < 1e-12 for a, b in zip(colormatrix.ColorMatrix(M).apply(C), C))),
    ("sat_zero", lambda: all(abs(a - b) < 1e-9 for a, b in zip(colormatrix.ColorMatrix(M).saturate(C, 0.0), (0.4569, 0.4569, 0.4569)))),
    ("sat_one", lambda: all(abs(a - b) < 1e-12 for a, b in zip(colormatrix.ColorMatrix(M).saturate(C, 1.0), C))),
    ("inverse", lambda: isinstance(colormatrix.ColorMatrix(M).inverse(), list)),
    ("bad_matrix", lambda: _raises(lambda: colormatrix.ColorMatrix([[1.0, 0.0], [0.0, 1.0]]))),
    ("singular", lambda: _raises(lambda: colormatrix.ColorMatrix([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]]).inverse())),
    ("clamp", lambda: all(0.0 <= v <= 1.0 for v in colormatrix.ColorMatrix([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).apply(C))),
    ("type", lambda: isinstance(colormatrix.ColorMatrix(M).apply(C), tuple)),
    ("det", lambda: colormatrix.ColorMatrix(M).apply(C) == colormatrix.ColorMatrix(M).apply(C)),
    ("nan", lambda: _raises(lambda: colormatrix.ColorMatrix(M).apply((float('nan'), 0.0, 0.0)))),
    ("no_mut", lambda: colormatrix.ColorMatrix([r[:] for r in M]).apply(C) is not None),
    ("sat_type", lambda: isinstance(colormatrix.ColorMatrix(M).saturate(C, 0.5)[0], float)),
    ("bad_s", lambda: _raises(lambda: colormatrix.ColorMatrix(M).saturate(C, -0.1))),
    ("range", lambda: all(0.0 <= v <= 1.0 for v in colormatrix.ColorMatrix(M).saturate(C, 0.5))),
    ("inverse_type", lambda: isinstance(colormatrix.ColorMatrix(M).inverse()[0], list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
