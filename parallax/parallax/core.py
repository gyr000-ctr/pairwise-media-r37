"""parallax 核心 API：陡峭视差贴图（steep parallax mapping）步进。

仅使用 Python 标准库。

高度场约定
----------
``height[row][col]`` 是取值在 ``[0, 1]`` 内的二维高度场：``row`` 对应 ``v``，
``col`` 对应 ``u``。UV 通过 ``(rows - 1, cols - 1)`` 线性映射到网格，采样前
越界坐标 clamp 到 ``[0, 1]``（边界 clamp）。

步进模型
--------
把高度场看作切线空间中的高程曲面 ``y = h(u, v)``，高度 ``y`` 从 ``0``（基准面）
到 ``1``（最高点）。观察者的水平视差向量为 ``view = (vx, vy)``：高度为 ``y``
的表面上某点，其基准 UV 为 ``uv + y * view``。于是从观察者射向表面的视线在
高度 ``y`` 处落在::

    P(y) = uv + y * view

把高度区间 ``[0, 1]`` 划分为 ``layers`` 层，从 ``y = 1`` 向 ``y = 0`` 逐层
采样高度图，找到第一个被表面挡住的层（``h(P(y)) > y``）。命中后先在相邻两层
之间线性插值，再二分细化，直到交点处的深度残差
``|h(P(y)) - y| <= (1 / layers) / 4``，故深度差恒 ``<= 1 / layers``（可测）。

若整条视线都没有被遮挡（全 0 高度场、平坦高度场、或沿线高度始终低于视线），
则回退到最近层——最靠近基准面的采样层 ``y = 0``，此时 ``P(0) = uv``，偏移为
``(0, 0)``。因此全 0 高度场的偏移严格为 ``0.0``。

``offset`` 返回修正量 ``(du, dv)``，且 ``uv + offset`` 会被 clamp 到 ``[0, 1]``。
``sample`` 返回修正后位置处的高度场值，同样位于 ``[0, 1]``。
"""

__all__ = ["Parallax"]

# 命中后二分细化的最大次数；残差进入容差即提前结束。
_MAX_REFINE = 12


def _as_pair(value, name):
    """把有限实数二元序列转成 ``(float, float)``；否则抛 ``ValueError``。

    ``(float, float)`` 元组走最快路径，其余类型再逐项校验。
    """
    if type(value) is tuple and len(value) == 2:
        first = value[0]
        second = value[1]
        if type(first) is float and type(second) is float:
            # inf-inf 与 nan-nan 均为 nan（!= 0.0），一并拒绝。
            if first - first != 0.0 or second - second != 0.0:
                raise ValueError("%s must be finite" % name)
            return first, second
    try:
        first = value[0]
        second = value[1]
        if len(value) != 2:
            raise ValueError("%s must have length 2" % name)
    except (TypeError, IndexError, KeyError):
        raise ValueError("%s must be a sequence of length 2" % name) from None
    if isinstance(first, bool) or not isinstance(first, (int, float)):
        raise ValueError("%s must contain real numbers" % name)
    if isinstance(second, bool) or not isinstance(second, (int, float)):
        raise ValueError("%s must contain real numbers" % name)
    first = float(first)
    second = float(second)
    if first - first != 0.0 or second - second != 0.0:
        raise ValueError("%s must be finite" % name)
    return first, second


def _sample_uv(H, u, v, xs, ys, xlast, ylast):
    """对 ``(u, v)`` clamp 到 ``[0, 1]`` 后做双线性采样。"""
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    fx = u * xs
    fy = v * ys
    ix = int(fx)
    if ix >= xlast:
        ix = xlast
        ix1 = xlast
        tx = 0.0
    else:
        ix1 = ix + 1
        tx = fx - ix
    iy = int(fy)
    if iy >= ylast:
        iy = ylast
        iy1 = ylast
        ty = 0.0
    else:
        iy1 = iy + 1
        ty = fy - iy
    r0 = H[iy]
    r1 = H[iy1]
    top = r0[ix] + (r0[ix1] - r0[ix]) * tx
    bottom = r1[ix] + (r1[ix1] - r1[ix]) * tx
    return top + (bottom - top) * ty


def _refine(H, u, v, vx, vy, xs, ys, xlast, ylast, y, y_prev, g, g_prev, inv):
    """在 ``[y, y_prev]`` 内定位 ``g(y) = 0`` 的交点并返回 UV 偏移。

    调用方保证 ``g > 0 >= g_prev``。先割线插值，再二分直至
    ``|g| <= inv / 4``（即深度残差小于四分之一层）。
    """
    tol = inv * 0.25
    lo_y = y          # g > 0 一侧
    hi_y = y_prev     # g <= 0 一侧
    denom = g_prev - g
    if denom > 0.0:
        y_hit = y_prev + (g_prev / denom) * (y - y_prev)
        if y_hit <= lo_y:
            y_hit = lo_y
        elif y_hit >= hi_y:
            y_hit = hi_y
    else:
        y_hit = 0.5 * (lo_y + hi_y)

    for _ in range(_MAX_REFINE):
        gm = _sample_uv(H, u + y_hit * vx, v + y_hit * vy, xs, ys, xlast, ylast) - y_hit
        if gm > 0.0:
            lo_y = y_hit
        else:
            hi_y = y_hit
        if -tol <= gm <= tol:
            break
        mid = 0.5 * (lo_y + hi_y)
        if mid == y_hit:  # 区间已收缩到浮点极限
            break
        y_hit = mid

    nu = u + y_hit * vx
    nv = v + y_hit * vy
    if nu < 0.0:
        nu = 0.0
    elif nu > 1.0:
        nu = 1.0
    if nv < 0.0:
        nv = 0.0
    elif nv > 1.0:
        nv = 1.0
    return nu - u, nv - v


def _scan_height(height):
    """校验高度场，返回 ``(rows, cols, hmin, hmax)``。

    非二维、空、非矩形、含非实数、越界或 NaN 的输入均抛 ``ValueError``。
    """
    try:
        rows = len(height)
    except TypeError:
        raise ValueError("height must be a 2D sequence") from None
    if rows == 0:
        raise ValueError("height must not be empty")
    try:
        cols = len(height[0])
    except (TypeError, IndexError, KeyError):
        raise ValueError("height must be a 2D sequence") from None
    if cols == 0:
        raise ValueError("height rows must not be empty")

    hmin = 1.0
    hmax = 0.0
    for row in height:
        try:
            if len(row) != cols:
                raise ValueError("height must be rectangular")
        except TypeError:
            raise ValueError("height must be a 2D sequence") from None
        try:
            for value in row:
                # 常见情况：float；int 与其它类型走慢路径校验。
                if type(value) is not float:
                    if isinstance(value, bool) or not isinstance(
                        value, (int, float)
                    ):
                        raise ValueError("height values must be real numbers")
                    value = float(value)
                # NaN 与任何数比较都是 False，故下面的判断同时拒绝 NaN。
                if not (0.0 <= value <= 1.0):
                    raise ValueError("height values must be within [0, 1]")
                if value < hmin:
                    hmin = value
                if value > hmax:
                    hmax = value
        except TypeError:
            raise ValueError("height must be a 2D sequence") from None
    return rows, cols, float(hmin), float(hmax)


class Parallax:
    """陡峭视差贴图步进器。"""

    __slots__ = ("layers", "_inv", "_cached", "_rows", "_cols", "_hmin", "_hmax")

    def __init__(self, layers):
        if isinstance(layers, bool) or not isinstance(layers, int):
            raise ValueError("layers must be an integer")
        if layers <= 0:
            raise ValueError("layers must be positive")
        self.layers = layers
        self._inv = 1.0 / layers
        # 校验结果缓存：按高度场对象身份（持有强引用，避免 id 复用）。
        # 采样始终读取调用方实时数据，缓存只用于跳过重复校验。
        self._cached = None
        self._rows = 0
        self._cols = 0
        self._hmin = 0.0
        self._hmax = 0.0

    # -- 校验 ---------------------------------------------------------
    def _info(self, height):
        if height is self._cached:
            return self._rows, self._cols, self._hmin, self._hmax
        rows, cols, hmin, hmax = _scan_height(height)
        self._cached = height
        self._rows = rows
        self._cols = cols
        self._hmin = hmin
        self._hmax = hmax
        return rows, cols, hmin, hmax

    # -- 公开 API -----------------------------------------------------
    @staticmethod
    def uv_clamped(uv):
        """把 UV clamp 到单位正方形 ``[0, 1] x [0, 1]``。"""
        u, v = _as_pair(uv, "uv")
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        return u, v

    def offset(self, uv, view, height):
        """返回把 ``uv`` 修正到可见表面点所需的二维偏移 ``(du, dv)``。

        返回值保证 ``uv + offset`` 位于 ``[0, 1]`` 内。
        """
        u, v = _as_pair(uv, "uv")
        vx, vy = _as_pair(view, "view")
        if vx == 0.0 and vy == 0.0:
            raise ValueError("view must be non-zero")
        rows, cols, hmin, hmax = self._info(height)

        if hmax <= 0.0:
            # 全 0 高度场：视线不可能被遮挡，严格零偏移。
            return 0.0, 0.0
        if hmin == hmax:
            # 平坦高度场：解析解（可见点固定在 y = h），无需步进。
            nu = u + hmin * vx
            nv = v + hmin * vy
            if nu < 0.0:
                nu = 0.0
            elif nu > 1.0:
                nu = 1.0
            if nv < 0.0:
                nv = 0.0
            elif nv > 1.0:
                nv = 1.0
            return nu - u, nv - v

        return self._march(u, v, vx, vy, height, rows, cols, hmax)

    def sample(self, uv, view, height):
        """返回可见表面点处的高度场采样值，位于 ``[0, 1]``。"""
        u, v = _as_pair(uv, "uv")
        vx, vy = _as_pair(view, "view")
        if vx == 0.0 and vy == 0.0:
            raise ValueError("view must be non-zero")
        rows, cols, hmin, hmax = self._info(height)

        if hmax <= 0.0:
            return 0.0
        if hmin == hmax:
            return hmin

        du, dv = self._march(u, v, vx, vy, height, rows, cols, hmax)
        nu = u + du
        nv = v + dv
        if nu < 0.0:
            nu = 0.0
        elif nu > 1.0:
            nu = 1.0
        if nv < 0.0:
            nv = 0.0
        elif nv > 1.0:
            nv = 1.0

        # 双线性采样修正后的位置。
        fx = nu * (cols - 1.0)
        fy = nv * (rows - 1.0)
        xlast = cols - 1
        ylast = rows - 1
        ix = int(fx)
        if ix >= xlast:
            ix = xlast
            ix1 = xlast
            tx = 0.0
        else:
            ix1 = ix + 1
            tx = fx - ix
        iy = int(fy)
        if iy >= ylast:
            iy = ylast
            iy1 = ylast
            ty = 0.0
        else:
            iy1 = iy + 1
            ty = fy - iy
        r0 = height[iy]
        r1 = height[iy1]
        top = r0[ix] + (r0[ix1] - r0[ix]) * tx
        bottom = r1[ix] + (r1[ix1] - r1[ix]) * tx
        value = top + (bottom - top) * ty
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return float(value)

    # -- 核心步进 -----------------------------------------------------
    def _march(self, u, v, vx, vy, height, rows, cols, hmax):
        layers = self.layers
        inv = self._inv
        xs = cols - 1.0
        ys = rows - 1.0

        # 整条视线在 [0, 1]^2 内时走无 clamp 的快路径。
        if vx >= 0.0:
            inside_u = u >= 0.0 and u + vx <= 1.0
        else:
            inside_u = u + vx >= 0.0 and u <= 1.0
        if vy >= 0.0:
            inside_v = v >= 0.0 and v + vy <= 1.0
        else:
            inside_v = v + vy >= 0.0 and v <= 1.0
        if inside_u and inside_v:
            return self._march_fast(u, v, vx, vy, height, rows, cols, hmax)
        return self._march_clamped(u, v, vx, vy, height, rows, cols, hmax)

    def _march_fast(self, u, v, vx, vy, height, rows, cols, hmax):
        """视线完全位于 ``[0, 1]^2`` 内，免去逐层边界判断。"""
        layers = self.layers
        inv = self._inv
        xs = cols - 1.0
        ys = rows - 1.0
        xlast = cols - 1
        ylast = rows - 1
        H = height

        # y >= hmax 的层必不可能命中，直接跳到第一个 y < hmax 的层。
        k0 = int(layers * (1.0 - hmax)) + 1
        if k0 < 1:
            k0 = 1
        elif k0 > layers:
            k0 = layers

        y = 1.0 - k0 * inv
        y_prev = y + inv
        fx = (u + y * vx) * xs
        fy = (v + y * vy) * ys
        sfx = inv * vx * xs
        sfy = inv * vy * ys
        g_prev = None

        for _ in range(k0, layers + 1):
            ix = int(fx)
            if ix >= xlast:
                ix = xlast
                tx = 0.0
                r0 = H[int(fy)]
                r1 = H[ylast]
                c0 = r0[xlast]
                c1 = r1[xlast]
                ty = fy - int(fy) if int(fy) < ylast else 0.0
                h = c0 + (c1 - c0) * ty
            else:
                iy = int(fy)
                if iy >= ylast:
                    ty = 0.0
                    r0 = H[ylast]
                    bottom = r0[ix] + (r0[ix + 1] - r0[ix]) * (fx - ix)
                    h = bottom
                else:
                    r0 = H[iy]
                    r1 = H[iy + 1]
                    nix = ix + 1
                    tx = fx - ix
                    ty = fy - iy
                    top = r0[ix] + (r0[nix] - r0[ix]) * tx
                    bottom = r1[ix] + (r1[nix] - r1[ix]) * tx
                    h = top + (bottom - top) * ty

            if h > y:
                if g_prev is None:
                    g_prev = (
                        _sample_uv(
                            H, u + y_prev * vx, v + y_prev * vy,
                            xs, ys, xlast, ylast,
                        )
                        - y_prev
                    )
                return _refine(
                    H, u, v, vx, vy, xs, ys, xlast, ylast,
                    y, y_prev, h - y, g_prev, inv,
                )

            g_prev = h - y
            y_prev = y
            y -= inv
            fx -= sfx
            fy -= sfy

        return 0.0, 0.0

    def _march_clamped(self, u, v, vx, vy, height, rows, cols, hmax):
        """通用路径：视线可能越出 ``[0, 1]^2``，逐层 clamp 后采样。"""
        layers = self.layers
        inv = self._inv
        xs = cols - 1.0
        ys = rows - 1.0
        xlast = cols - 1
        ylast = rows - 1
        H = height

        k0 = int(layers * (1.0 - hmax)) + 1
        if k0 < 1:
            k0 = 1
        elif k0 > layers:
            k0 = layers

        y = 1.0 - k0 * inv
        y_prev = y + inv
        pu = u + y * vx
        pv = v + y * vy
        step_u = inv * vx
        step_v = inv * vy
        g_prev = None

        for _ in range(k0, layers + 1):
            qu = pu
            if qu < 0.0:
                qu = 0.0
            elif qu > 1.0:
                qu = 1.0
            qv = pv
            if qv < 0.0:
                qv = 0.0
            elif qv > 1.0:
                qv = 1.0
            fx = qu * xs
            fy = qv * ys
            ix = int(fx)
            if ix >= xlast:
                ix = xlast
                ix1 = xlast
                tx = 0.0
            else:
                ix1 = ix + 1
                tx = fx - ix
            iy = int(fy)
            if iy >= ylast:
                iy = ylast
                iy1 = ylast
                ty = 0.0
            else:
                iy1 = iy + 1
                ty = fy - iy
            r0 = H[iy]
            r1 = H[iy1]
            top = r0[ix] + (r0[ix1] - r0[ix]) * tx
            bottom = r1[ix] + (r1[ix1] - r1[ix]) * tx
            h = top + (bottom - top) * ty

            if h > y:
                if g_prev is None:
                    g_prev = (
                        _sample_uv(
                            H, u + y_prev * vx, v + y_prev * vy,
                            xs, ys, xlast, ylast,
                        )
                        - y_prev
                    )
                return _refine(
                    H, u, v, vx, vy, xs, ys, xlast, ylast,
                    y, y_prev, h - y, g_prev, inv,
                )
            g_prev = h - y
            y_prev = y
            y -= inv
            pu -= step_u
            pv -= step_v

        return 0.0, 0.0
