"""tonehable 核心 API（初始未实现，仅标准库）。"""


class Hable:
    def curve(self, x):
        """前向"""
        raise NotImplementedError

    def inverse(self, y):
        """反解"""
        raise NotImplementedError

    def white(self):
        """白点"""
        raise NotImplementedError
