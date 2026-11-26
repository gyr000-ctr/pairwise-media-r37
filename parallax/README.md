# parallax

## 题面（可直接粘贴）

视差贴图步进（陡峭视差）。约束：步进层级可配且按高度图采样，命中深度差 ≤1/层数（可测）；未命中时回退到最近层并与线性搜索对照 ≤1 层；边界 clamp；高度为 0 时不偏移（≤1e-12）；10 万次不超过 400 ms。 工作目录为仓库内的 parallax/。 验收：cd parallax && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd parallax && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
