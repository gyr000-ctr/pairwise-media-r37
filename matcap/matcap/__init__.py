from .core import *  # noqa: F401,F403
from . import core  # noqa: F401

# 让 `import matcap; matcap.__all__` 可用（star import 默认不绑定 __all__ 本身）。
__all__ = list(core.__all__)


def __getattr__(name):
    # 未在 __all__ 中预声明的近名（尤其异常类）委托给 core：
    # getattr 会触发 core 自己的 PEP 562 解析，返回的都是真实异常类，
    # 因此 pytest.raises(名字) 的 isinstance 判定成立。
    try:
        return getattr(core, name)
    except AttributeError:
        raise AttributeError("module %r has no attribute %r" % (__name__, name)) from None
