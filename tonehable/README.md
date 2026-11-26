# tonehable

## 题面（可直接粘贴）

Hable 色调映射与逆。约束：曲线单调且 0→0；与参考系数（A/B/C/D/E/F）逐项一致 ≤1e-12；必须提供反解且往返 ≤1e-6（牛顿迭代）；白点归一后 11.2 映射到 1（≤1e-6）；10 万次求值不超过 150 ms。 工作目录为仓库内的 tonehable/。 验收：cd tonehable && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd tonehable && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
