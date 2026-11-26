"""colormatrix 核心 API（初始未实现，仅标准库）。"""


class ColorMatrix:
    def __init__(self, matrix):
        """构造"""
        raise NotImplementedError

    def apply(self, rgb):
        """应用"""
        raise NotImplementedError

    def saturate(self, rgb, s):
        """饱和度"""
        raise NotImplementedError
