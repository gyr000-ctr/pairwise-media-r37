# colormatrix

## 题面（可直接粘贴）

色彩矩阵、饱和度与矩阵分解。约束：3x3 矩阵逐通道与解析式一致（≤1e-12）；s=0 输出 Rec.709 亮度、s=1 恒等（≤1e-12）；必须实现 LU 分解求逆，并与伴随矩阵法对照（≤1e-9）；条件数估计与解析值差 ≤1e-9，奇异阈值可配且超阈报错；越界钳制到 [0,1]；100 万像素不超过 500 ms。 工作目录为仓库内的 colormatrix/。 验收：cd colormatrix && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd colormatrix && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
