"""双边滤波（空间核与值域核分离的可分离实现，仅使用 Python 标准库）。

- 空间核：一维高斯，仅取决于像素偏移；先水平后垂直两遍应用（可分离）。
- 值域核：高斯，仅取决于像素值与中心像素值之差，逐像素应用。
- 每个像素的权重都按实际参与累加的权重归一化，权重和恒为 1，
  因此常量图在滤波后保持不变。
- 大图（百万像素级）自动使用 multiprocessing 按行块并行；并行不可用时
  回退到串行实现，结果完全一致（每行独立计算，按原顺序拼接）。
"""

import math
import multiprocessing

__all__ = ["Bilateral"]

# 超过该工作量（像素数 × 每像素 tap 数）才启用并行，小图走串行避免进程开销
_PAR_MIN_TAPS = 2_000_000
_MAX_WORKERS = 8


def _check_sigma(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "%s must be a real number, got %s" % (name, type(value).__name__)
        )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "%s must be a positive finite number, got %r" % (name, value)
        )
    return float(value)


def _smooth_rows(rows, spatial, ninv, radius, clamp=None):
    """对每一行做 1D 双边滤波（空间核 spatial，值域系数 ninv = -1/(2*sigma_r^2)）。"""
    r = radius
    sp = spatial
    exp = math.exp
    n = 2 * r + 1
    out = []
    for row in rows:
        w = len(row)
        orow = [0.0] * w
        if w > n:
            # 内部像素：窗口完整，无需边界检查
            for x in range(r, w - r):
                c = row[x]
                num = 0.0
                den = 0.0
                for sk, v in zip(sp, row[x - r:x + r + 1]):
                    d = v - c
                    wt = sk * exp(d * d * ninv)
                    num += wt * v
                    den += wt
                orow[x] = num / den
            xs = list(range(0, r)) + list(range(w - r, w))
        else:
            xs = range(w)
        # 边缘像素：窗口被裁断，权重按实际参与部分归一化
        for x in xs:
            a = x - r
            b = x + r
            k0 = 0
            if a < 0:
                k0 = -a
                a = 0
            if b >= w:
                b = w - 1
            c = row[x]
            num = 0.0
            den = 0.0
            for sk, v in zip(sp[k0:k0 + (b - a + 1)], row[a:b + 1]):
                d = v - c
                wt = sk * exp(d * d * ninv)
                num += wt * v
                den += wt
            orow[x] = num / den
        if clamp is not None:
            lo, hi = clamp
            orow = [lo if v < lo else hi if v > hi else v for v in orow]
        out.append(orow)
    return out


def _smooth_star(args):
    """multiprocessing 回调（顶层函数，可 pickle）。"""
    rows, spatial, ninv, radius, clamp = args
    return _smooth_rows(rows, spatial, ninv, radius, clamp)


def _chunks(seq, n):
    """把序列切成至多 n 个连续块，保持顺序。"""
    n = max(1, min(n, len(seq)))
    q, rem = divmod(len(seq), n)
    out = []
    i = 0
    for c in range(n):
        j = i + q + (1 if c < rem else 0)
        if j > i:
            out.append(seq[i:j])
        i = j
    return out


class Bilateral:
    """双边滤波器。

    sigma_s — 空间域标准差（正有限数）；sigma_r — 值域标准差（正有限数）；
    radius — 核半径（正整数）。sigma 越大，平滑越强。
    """

    def __init__(self, sigma_s, sigma_r, radius):
        self.sigma_s = _check_sigma("sigma_s", sigma_s)
        self.sigma_r = _check_sigma("sigma_r", sigma_r)
        if isinstance(radius, bool) or not isinstance(radius, int):
            raise TypeError(
                "radius must be an integer, got %s" % type(radius).__name__
            )
        if radius < 1:
            raise ValueError("radius must be a positive integer, got %r" % (radius,))
        self.radius = radius
        inv2s2 = 1.0 / (2.0 * self.sigma_s * self.sigma_s)
        # 空间核（一维），与值域核完全分离
        self._spatial = [
            math.exp(-(k * k) * inv2s2) for k in range(-radius, radius + 1)
        ]
        self._neg_inv2r2 = -1.0 / (2.0 * self.sigma_r * self.sigma_r)

    def kernel(self, center=0.0):
        """返回以 center 为中心的 2D 空间核（按行展开的扁平列表，权重和为 1）。

        值域核依赖邻域实际像素值，在 apply() 中逐像素应用；对常量邻域
        （所有像素都等于 center）值域权重恒为 1，此时该核即完整双边核。
        """
        if isinstance(center, bool) or not isinstance(center, (int, float)):
            raise TypeError(
                "center must be a real number, got %s" % type(center).__name__
            )
        if not math.isfinite(center):
            raise ValueError("center must be finite, got %r" % (center,))
        sp = self._spatial
        n = 2 * self.radius + 1
        w = [sp[dy] * sp[dx] for dy in range(n) for dx in range(n)]
        total = math.fsum(w)
        return [x / total for x in w]

    def apply(self, image):
        """对灰度图（非空列表的列表）做双边滤波，返回新的列表的列表。"""
        rows, lo, hi = self._validate(image)
        taps = len(rows) * len(rows[0]) * (2 * self.radius + 1)
        if taps >= _PAR_MIN_TAPS:
            try:
                return self._apply_parallel(rows, lo, hi)
            except Exception:
                pass  # 并行环境不可用时回退串行
        tmp = _smooth_rows(rows, self._spatial, self._neg_inv2r2, self.radius)
        out_cols = _smooth_rows(
            list(zip(*tmp)), self._spatial, self._neg_inv2r2, self.radius,
            clamp=(lo, hi),
        )
        return [list(row) for row in zip(*out_cols)]

    # ---- 内部实现 ----

    def _apply_parallel(self, rows, lo, hi):
        ctx = multiprocessing.get_context("fork")
        nproc = min(multiprocessing.cpu_count(), _MAX_WORKERS)
        common = (self._spatial, self._neg_inv2r2, self.radius)
        with ctx.Pool(nproc) as pool:
            tmp_parts = pool.map(
                _smooth_star,
                [(chunk,) + common + (None,) for chunk in _chunks(rows, nproc)],
            )
            tmp = [row for part in tmp_parts for row in part]
            cols = list(zip(*tmp))
            out_parts = pool.map(
                _smooth_star,
                [(chunk,) + common + ((lo, hi),) for chunk in _chunks(cols, nproc)],
            )
            out_cols = [row for part in out_parts for row in part]
        return [list(row) for row in zip(*out_cols)]

    @staticmethod
    def _validate(image):
        if not isinstance(image, (list, tuple)) or len(image) == 0:
            raise ValueError("image must be a non-empty list of rows")
        width = None
        for row in image:
            if not isinstance(row, (list, tuple)) or len(row) == 0:
                raise ValueError("image rows must be non-empty lists of numbers")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError("image rows must all have the same length")
        finite = math.isfinite
        try:
            ok = all(finite(v) for row in image for v in row)
        except TypeError:
            raise TypeError("pixel values must be real numbers") from None
        if not ok:
            raise ValueError("pixel values must be finite (no NaN or inf)")
        lo = min(min(row) for row in image)
        hi = max(max(row) for row in image)
        return image, lo, hi
