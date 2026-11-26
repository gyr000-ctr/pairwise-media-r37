"""Hable 色调映射与逆（仅使用 Python 标准库）。

前向采用 John Hable 的 filmic 曲线::

    f(x) = (x * (A*x + C*B) + D*E) / (x * (A*x + B) + D*F) - E/F

其中参考系数为 A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30。
``curve`` 以 ``f(white)`` 归一化并钳制到 [0, 1]；``inverse`` 先解析解出
等价的二次方程作为初值，再用牛顿迭代收敛到机器精度。
"""

import math

__all__ = ["Hable", "A", "B", "C", "D", "E", "F"]

# 参考系数（Hable filmic tone mapping）。
A = 0.15
B = 0.50
C = 0.10
D = 0.20
E = 0.02
F = 0.30

# 预计算常量，避免每次求值重复计算。
_DF = D * F

# f 的无相消形式 f(x) = x*(p*x + q) / (F*(A*x^2 + B*x + D*F))：
#   F*(A x^2 + C B x + D E) - E*(A x^2 + B x + D F)
#     = x * (A(F-E) x + B(F C - E))，因为 F*D*E == E*D*F。
_P = A * (F - E)
_Q = B * (F * C - E)


class Hable:
    """Hable filmic 色调映射。

    Parameters
    ----------
    white:
        白点亮度（必须为正的有限数）。归一化后 ``curve(white) == 1``。
    """

    #: 参考系数，便于外部逐项核对。
    A = A
    B = B
    C = C
    D = D
    E = E
    F = F

    __slots__ = ("_white", "_norm", "_inv_norm")

    def __init__(self, white=11.2):
        if isinstance(white, bool) or not isinstance(white, (int, float)):
            raise ValueError("white 必须是实数")
        w = float(white)
        if not math.isfinite(w) or w <= 0.0:
            raise ValueError(f"white 必须是正的有限数，收到 {white!r}")
        self._white = w
        # f 在 (0, +inf) 上严格递增，白点处为正。
        norm = self._raw(w)
        self._norm = norm
        self._inv_norm = 1.0 / norm

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    @staticmethod
    def _raw(x):
        """未归一化的 Hable 曲线值（无相消形式）。"""
        if x >= 1.0:
            # 以 u = 1/x 展开，避免 x 很大时溢出（结果等价）。
            u = 1.0 / x
            num = _P + _Q * u
            den = F * (A + B * u + _DF * u * u)
            return num / den
        return x * (_P * x + _Q) / (F * (A * x * x + B * x + _DF))

    @staticmethod
    def _raw_with_derivative(x):
        """未归一化曲线值及其导数（牛顿迭代用）。"""
        if x >= 1.0:
            # p(u) = N(u)/D(u)，u = 1/x，N = _P + _Q*u，D = F*(A + B*u + DF*u^2)。
            # dp/dx = -u^2 * (N'D - ND') / D^2。
            u = 1.0 / x
            u2 = u * u
            num = _P + _Q * u
            den = F * (A + B * u + _DF * u2)
            # N'D - ND' = F * (Q*A - P*B - 2*P*DF*u - Q*DF*u^2)
            bracket = F * (_Q * A - _P * B - 2.0 * _P * _DF * u - _Q * _DF * u2)
            return num / den, -u2 * bracket / (den * den)
        xx = x * x
        num = x * (_P * x + _Q)
        den = F * (A * xx + B * x + _DF)
        dnum = 2.0 * _P * x + _Q
        dden = F * (2.0 * A * x + B)
        return num / den, (dnum * den - num * dden) / (den * den)

    def curve(self, x):
        """前向映射：返回 [0, 1] 内的浮点数（单调不减，0 -> 0）。"""
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"x 必须是实数，收到 {type(x).__name__}")
        xf = float(x)
        if xf != xf or xf < 0.0:
            raise ValueError(f"x 必须是非负实数，收到 {x!r}")
        if xf == 0.0:
            return 0.0
        # f 严格递增且 f(white) 是归一化基准，故 x >= white 时结果恒为 1。
        if xf >= self._white:
            return 1.0
        value = self._raw(xf) * self._inv_norm
        if value <= 0.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        return value

    # ------------------------------------------------------------------
    # 反解
    # ------------------------------------------------------------------
    def inverse(self, y):
        """前向的反解：解析二次初值 + 牛顿迭代，定义域 [0, 1]。"""
        if isinstance(y, bool) or not isinstance(y, (int, float)):
            raise ValueError(f"y 必须是实数，收到 {type(y).__name__}")
        yf = float(y)
        if yf != yf or yf < 0.0 or yf > 1.0:
            raise ValueError(f"y 必须位于 [0, 1]，收到 {y!r}")
        if yf == 0.0:
            return 0.0
        if yf == 1.0:
            return self._white

        target = yf * self._norm
        x = self._quadratic_guess(target)
        if x < 0.0:
            x = 0.0

        # 牛顿迭代：f(x) = target（归一化前后同一方程）。收敛到 x 的浮点极限
        # （残差已无法再减小）即停止，通常 2~6 步。
        best_x = x
        best_res = math.inf
        for _ in range(16):
            value, derivative = self._raw_with_derivative(x)
            res = value - target
            if abs(res) < best_res:
                best_res = abs(res)
                best_x = x
            else:
                # 残差不再下降：已到浮点极限，取最优点即可。
                break
            if derivative <= 0.0:
                break
            nxt = x - res / derivative
            if nxt < 0.0:
                nxt = 0.0
            if nxt == x:
                break
            x = nxt
        x = best_x
        if x > self._white:
            x = self._white
        return float(x)

    @staticmethod
    def _quadratic_guess(z):
        """解 f(x) = z 的等价二次方程，取唯一的非负根作为牛顿初值。

        交叉相乘后：``p*x^2 + q*x + r = 0``，其中

            p = A*(F*(1 - z) - E)
            q = B*(F*(C - z) - E)
            r = -D*F^2*z

        ``r <= 0`` 而 ``p >= 0``（z 不超过曲线的值域上界），故两根之积非正，
        恰有一个非负根。按 q 的符号选择数值稳定的求根公式。
        """
        p = A * (F * (1.0 - z) - E)
        q = B * (F * (C - z) - E)
        r = -D * F * F * z
        if p == 0.0:
            return -r / q if q != 0.0 else 0.0
        disc = q * q - 4.0 * p * r
        if disc < 0.0:
            return 0.0
        sq = math.sqrt(disc)
        if q <= 0.0:
            return (-q + sq) / (2.0 * p)
        denom = -q - sq
        if denom == 0.0:
            return 0.0
        return (2.0 * r) / denom

    def white(self):
        """白点亮度。"""
        return self._white
