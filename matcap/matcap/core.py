"""matcap 核心 API（初始未实现，仅标准库）。"""


class MatCap:
    def __init__(self, matcap):
        """构造"""
        raise NotImplementedError

    def uv(self, normal):
        """法线到 UV"""
        raise NotImplementedError

    def sample(self, normal):
        """采样"""
        raise NotImplementedError
