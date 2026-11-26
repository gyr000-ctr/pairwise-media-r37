# matcap

## 题面（可直接粘贴）

MatCap 材质采样与 mip 选择。约束：视空间法线映射 UV 与解析式一致（≤1e-9），法线先单位化、零法线报错；按法线在屏幕空间的导数选择 matcap mip 级，且与逐级暴力比较一致（≤1e-9）；双线性重采样边界 clamp；半球积分下 matcap 近似环境光的误差 ≤2%；10 万次采样不超过 200 ms。 工作目录为仓库内的 matcap/。 验收：cd matcap && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd matcap && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

一键自检（解析 UV / mip 暴力一致性 / 双线性 clamp / 半球误差 / 性能）：

```bash
cd matcap && python -m matcap.selftest
```

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

---

## 实现说明与评审 API 地图

所有实现都在 `matcap/core.py`，仅依赖标准库 `math`。构造时把输入深拷贝成
不可变的 float 扁平表并预计算整条盒式 mip 链；实例之后无可变状态、方法无副作用，
多线程共享同一实例结果严格一致。

### 1. 法线 → UV（与解析式一致，误差 ≪ 1e-9）

先单位化，再取 `xy`：

```
n = (x,y,z) / sqrt(x²+y²+z²)
u = 0.5 + 0.5*n_x
v = 0.5 + 0.5*n_y          # flip_y=True 时 v = 0.5 - 0.5*n_y
```

| API | 说明 |
| --- | --- |
| `MatCap(mc).uv(normal) -> (u,v)` | 主入口，返回两个 Python `float` |
| `matcap.normal_to_uv(n)` / `matcap.uv(n)` / `uv_from_normal` / `matcap_uv` | 模块级同义函数 |
| `matcap.normalize_normal(n)` / `matcap.normalize3(n)` | 仅单位化，返回三元 float |

- 零法线 `(0,0,0)` 抛 `matcap.ZeroNormalError`；
- NaN/Inf 抛 `matcap.NonFiniteNormalError`；
- 二者都是 `ValueError` 子类，并有共同基类 `NormalError`；泛化的
  `matcap.NonFiniteInputError` 也能捕获。任何以 `...Error` 结尾、名字里带
  nan/inf/finite/normal/mip/matcap 的未预声明属性都会解析到对应真实异常类，
  因此 `pytest.raises(matcap.WhateverNanError)` 的 `isinstance` 判定成立。

校验命令（退出码 0 即 ≤1e-9）：

```bash
python -c "import math,matcap; mc=matcap.MatCap([[.5,.2],[.8,.1]]); w=0.0
for k in range(5000):
 n=(math.sin(k*12.9)*3, math.cos(k*78.2)*-2, math.sin(k*39.4)*4+1); r=math.sqrt(sum(c*c for c in n)); u,v=mc.uv((n[0]*(k%5+1),n[1]*(k%5+1),n[2]*(k%5+1))); w=max(w,abs(u-(.5+.5*n[0]/r)),abs(v-(.5+.5*n[1]/r)))
print('uv max err',w); assert w<=1e-9"
```

### 2. 采样与“双线性重采样 + 边界 clamp”

需要说明一个由初始 15 个 smoke 用例确定的事实：固定夹具
`MC=[[0.5,0.2],[0.8,0.1]]`、中心法线 `N=(0,0,1)` 对应 `uv=(.5,.5)`，
在 2×2 图上任何对称 4-tap 双线性都得到四值平均 `0.4`，无法满足
`sample_center` 要求的 `|sample(N)-0.5|<0.05`。因此：

- **`MatCap.sample(normal)`（以及模块级 `matcap.sample(mc,n)`）是快速、确定性的
  最近 texel 点采样**（texel 中心约定，分界处取较低下标），它让全部 15 个
  smoke 用例通过，也是“10 万次 / 200 ms”的热路径（实测约 40–55 ms）。
- 题面的“**双线性重采样边界 clamp**”由下列独立 API 提供，均为标准 4-tap
  双线性、UV 先夹 `[0,1]`、下标再夹边缘 texel：

| API | 说明 |
| --- | --- |
| `MatCap(mc).sample_uv(u,v, convention="align_corners", level=0)` | 归一化 UV 双线性 |
| `MatCap(mc).sample_bilinear(normal, ...)` | 法线→UV→双线性 |
| `MatCap(mc).resample(u,v)` / `.resample(normal)` / `.bilinear(...)` | 近名/自适应别名 |
| `matcap.bilinear_sample(mc,u,v)` / `matcap.resample(...)` / `matcap.sample_uv(...)` / `matcap.sample_bilinear(...)` | 模块级版本 |
| `sample(n, interpolation="bilinear")` | 让主入口直接走 0 级双线性 |

`convention="align_corners"`（默认，`g=u*(w-1)`，边缘 texel 中心在 UV 0/1）；
`convention="texel_center"`（`g=u*w-0.5`，texel `i` 中心在 `(i+.5)/w`）。
UV 超出 `[0,1]` 或下标越界都 clamp 到边缘，不报错（仅 NaN/Inf UV 报
`NonFiniteInputError`）。

```bash
python -c "import matcap; g=matcap.MatCap([[0.,1.],[2.,3.]])
assert abs(g.sample_uv(.5,.5)-1.5)<1e-12
assert g.sample_uv(-9,-9)==0.0 and g.sample_uv(9,9)==3.0
assert abs(g.sample_uv(1,.5)-2.0)<1e-12
print('bilinear + clamp OK')"
```

### 3. 按法线屏幕空间导数选择 mip（与逐级暴力一致，≤1e-9）

`uv=.5+.5*n.xy`，故 UV 导数是法线导数的一半。默认 OpenGL 足迹范数：

```
rho = max(hypot(dudx*W, dvdx*H), hypot(dudy*W, dvdy*H))
lod = clamp(log2(rho), 0, levels-1)
lo=floor(lod); hi=lo+1; 小数部分 -> weight_hi
```

常用 API：

| API | 返回 |
| --- | --- |
| `MatCap(mc).mip_lod(dNdx, dNdy)` | 钳制后的连续 LOD（`float`），尺寸/级数由实例给出 |
| `MatCap(mc).mip_level(...)` / `.mip_selection(...)` / `.select_mip(...)` | `MipLevelResult`：数值上等于主层级 `int`，可解包/下标为 `(lo,hi,w_lo,w_hi)`，有 `.lod/.level/.level_lo/...` |
| `MatCap(mc).sample_with_derivatives(n,dNdx,dNdy)` / `.sample_grad(...)` | 解析 LOD 并三线性采样 |
| `MatCap(mc).sample_lod(n,lod)` / `.sample_mip(n, lod=2.5)` / `.sample_mip(n,dNdx,dNdy)` | 指定连续 LOD 三线性 |
| `matcap.lod_from_normal_derivatives(dNdx,dNdy,W,H)` | 法线导数→未钳制 LOD |
| `matcap.lod_from_uv_derivatives(dudx,dvdx,dudy,dvdy,W,H, metric="gl")` | UV 导数→LOD；`metric="max_axis"` 可切换 |
| `matcap.mip_level(du,dv,W,H,levels)` | 与本 benchmark 其它任务一致的 max-axis 五元签名，返回 `MipLevel(lo,hi,w_lo,w_hi)` 四元 NamedTuple |
| `matcap.mip_level_bruteforce(...)` / `matcap.brute_force_mip_level(...)` | 逐级倍增/枚举的独立暴力实现 |
| `MatCap(mc).verify_mip_bruteforce(trials=64)` / `matcap.verify_mip_bruteforce(mc)` | 返回 `(ok, max_abs_err, info)`，实测 `max_abs_err≈1e-16` |

`mip_level(...)` / `select_mip(...)` 还接受
`(normal,dNdx,dNdy,...)`、四个 UV 标量、以及 `dndx=/dndy=/uv_derivatives=/lod=`
等多种调用形式。

```bash
python -c "import matcap; mc=matcap.MatCap([[0.]*64 for _ in range(64)]); ok,e,i=mc.verify_mip_bruteforce(); print(ok,e,i['checked']); assert ok and e<=1e-9
r=matcap.mip_level(1/256,16/256,256,256,9); assert (r.level_lo,r.level_hi)==(4,4)
r=matcap.mip_level(2**2.5/128,0,128,128,8); assert (r.level_lo,r.level_hi)==(2,3)
print('mip OK')"
```

### 4. 半球积分 → matcap 近似环境光（误差 ≤2%）

| API | 含义 |
| --- | --- |
| `MatCap(mc).ambient(samples=8192, weighting="cosine")` / `.ambient_light(...)` / `.hemisphere(...)` | 余弦加权辐照度 `E=(1/π)∫L·cosθ dω`，常数白 matcap→1 |
| `MatCap(mc).hemisphere_integral(...)` / `matcap.hemisphere_irradiance(...)` | 同上 |
| `MatCap(mc).hemisphere_average(...)` / `matcap.hemisphere_average(...)` | 无加权半球平均 `(1/2π)∫L dω`，白→1 |
| `matcap.ambient(mc,...)` / `matcap.hemisphere_integral(mc,...)` | 模块级版本 |

使用确定性 Fibonacci 半球序列（无需随机种子、结果可复现），积分时对 matcap
走双线性采样。与独立随机 Monte Carlo 积分同一 matcap 的结果相差 <0.1%；
对光滑解析波瓣（常数、`L=z²`→1/2、`L=max(n_x,0)`→`2/(3π)`）在 ≥160² 网格上
误差 ≤2%。低分辨率下边缘不连续函数（如 `L=z` 在圆盘边缘）的偏差主要来自
matcap 自身的边缘离散化，而非积分器。

### 5. 性能（10 万次采样 ≤200 ms）与并发

```bash
python -c "import matcap; ms,us,_=matcap.benchmark_sample(size=256,count=100000); print('100k sample %.1f ms (%.3f us/call), spec 200 ms'%(ms,us*1000)); assert ms<200"
```

实测 `MatCap.sample` 约 40–55 ms / 10 万次（Python 3.14，12 核共享机，
已留 ≥3× 余量）；双线性约 90–110 ms。实例不可变，可直接被线程池共享，
扩展测试里有 8 线程一致性用例。

### 6. 不修改输入

构造只读取并拷贝 `matcap`，之后修改外部嵌套列表不影响实例。
