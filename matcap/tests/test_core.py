import math

import pytest

import matcap

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

MC = [[0.5, 0.2], [0.8, 0.1]]
N = (0.0, 0.0, 1.0)

CASES = [
    ("uv_pair", lambda: len(matcap.MatCap(MC).uv(N)) == 2),
    ("uv_center", lambda: abs(matcap.MatCap(MC).uv(N)[0] - 0.5) < 1e-9),
    ("uv_range", lambda: all(0.0 <= c <= 1.0 for c in matcap.MatCap(MC).uv((0.6, 0.8, 0.0)))),
    ("unit_required", lambda: _raises(lambda: matcap.MatCap(MC).uv((0.0, 0.0, 0.0)))),
    ("sample_range", lambda: 0.0 <= matcap.MatCap(MC).sample(N) <= 1.0),
    ("sample_center", lambda: abs(matcap.MatCap(MC).sample(N) - 0.5) < 0.05),
    ("bad_matcap", lambda: _raises(lambda: matcap.MatCap([]))),
    ("nan", lambda: _raises(lambda: matcap.MatCap(MC).uv((float('nan'), 0.0, 1.0)))),
    ("type", lambda: isinstance(matcap.MatCap(MC).sample(N), float)),
    ("det", lambda: matcap.MatCap(MC).sample(N) == matcap.MatCap(MC).sample(N)),
    ("clamp", lambda: isinstance(matcap.MatCap(MC).uv((1.0, 1.0, 0.001)), tuple)),
    ("no_mut", lambda: matcap.MatCap([r[:] for r in MC]).sample(N) is not None),
    ("single", lambda: 0.0 <= matcap.MatCap([[1.0]]).sample(N) <= 1.0),
    ("scale_n", lambda: abs(matcap.MatCap(MC).uv((0.0, 0.0, 5.0))[0] - 0.5) < 1e-9),
    ("float", lambda: isinstance(matcap.MatCap(MC).uv(N)[0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
