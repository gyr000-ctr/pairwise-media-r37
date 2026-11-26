"""parallax 核心 API（初始未实现，仅标准库）。"""


class Parallax:
    def __init__(self, layers):
        """构造"""
        raise NotImplementedError

    def offset(self, uv, view, height):
        """偏移量"""
        raise NotImplementedError

    def sample(self, uv, view, height):
        """采样"""
        raise NotImplementedError
