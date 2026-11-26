import math

import pytest

import parallax

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

H = [[0.5, 0.5], [0.5, 0.5]]
V = (0.3, 0.4)

CASES = [
    ("offset_pair", lambda: len(parallax.Parallax(8).offset((0.5, 0.5), V, H)) == 2),
    ("zero_height", lambda: all(abs(c) < 1e-12 for c in parallax.Parallax(8).offset((0.5, 0.5), V, [[0.0, 0.0], [0.0, 0.0]]))),
    ("bad_layers", lambda: _raises(lambda: parallax.Parallax(0))),
    ("bad_view", lambda: _raises(lambda: parallax.Parallax(8).offset((0.5, 0.5), (0.0, 0.0), H))),
    ("sample", lambda: isinstance(parallax.Parallax(8).sample((0.5, 0.5), V, H), float)),
    ("bad_height", lambda: _raises(lambda: parallax.Parallax(8).offset((0.5, 0.5), V, []))),
    ("type", lambda: isinstance(parallax.Parallax(8).offset((0.5, 0.5), V, H), tuple)),
    ("det", lambda: parallax.Parallax(8).offset((0.5, 0.5), V, H) == parallax.Parallax(8).offset((0.5, 0.5), V, H)),
    ("layers_monotone", lambda: isinstance(parallax.Parallax(16).offset((0.2, 0.8), V, H), tuple)),
    ("nan", lambda: _raises(lambda: parallax.Parallax(8).offset((float('nan'), 0.5), V, H))),
    ("no_mut", lambda: parallax.Parallax(8).sample((0.3, 0.3), V, [r[:] for r in H]) is not None),
    ("clamp", lambda: all(0.0 <= c <= 1.0 for c in parallax.Parallax(8).uv_clamped((-0.5, 1.5)))),
    ("float", lambda: isinstance(parallax.Parallax(8).offset((0.2, 0.2), V, H)[0], float)),
    ("sample_range", lambda: 0.0 <= parallax.Parallax(8).sample((0.5, 0.5), V, H) <= 1.0),
    ("height_range", lambda: _raises(lambda: parallax.Parallax(8).offset((0.5, 0.5), V, [[2.0]]))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
