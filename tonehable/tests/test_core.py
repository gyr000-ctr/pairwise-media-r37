import math

import pytest

import tonehable

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _h():
    return tonehable.Hable(11.2)

CASES = [
    ("zero", lambda: _h().curve(0.0) == 0.0),
    ("monotone", lambda: _h().curve(0.3) <= _h().curve(0.8)),
    ("white_norm", lambda: abs(_h().curve(_h().white()) - 1.0) < 1e-6),
    ("inverse", lambda: abs(_h().inverse(_h().curve(0.4)) - 0.4) < 1e-6),
    ("bad_white", lambda: _raises(lambda: tonehable.Hable(0.0))),
    ("bad_negative", lambda: _raises(lambda: _h().curve(-0.1))),
    ("bad_inverse", lambda: _raises(lambda: _h().inverse(1.5))),
    ("type", lambda: isinstance(_h().curve(0.5), float)),
    ("det", lambda: _h().curve(0.5) == _h().curve(0.5)),
    ("white_type", lambda: isinstance(_h().white(), float)),
    ("bounded", lambda: _h().curve(100.0) <= 1.0),
    ("inverse_type", lambda: isinstance(_h().inverse(0.5), float)),
    ("nan", lambda: _raises(lambda: _h().curve(float('nan')))),
    ("no_mut", lambda: _h().curve(0.2) is not None),
    ("range", lambda: 0.0 <= _h().curve(0.7) <= 1.0),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
