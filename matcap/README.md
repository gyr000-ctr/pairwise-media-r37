# matcap

## 题面（可直接粘贴）

MatCap 材质采样与 mip 选择。约束：视空间法线映射 UV 与解析式一致（≤1e-9），法线先单位化、零法线报错；按法线在屏幕空间的导数选择 matcap mip 级，且与逐级暴力比较一致（≤1e-9）；双线性重采样边界 clamp；半球积分下 matcap 近似环境光的误差 ≤2%；10 万次采样不超过 200 ms。 工作目录为仓库内的 matcap/。 验收：cd matcap && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd matcap && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
