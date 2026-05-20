# astock-quant —— A股 量化预测系统（学习/研究用）

一个**从零搭建、用于学习与研究**的 A股 量化预测系统。代码追求可读、可跑、可改 —— 不是实盘交易系统。

> **当前状态：Stage 1 完成** ✅ —— 数据 / 因子 / 模型 / 回测 / 信号 全部跑通；47+ 测试全过；4 份人话报告全交付；待 Stage 2 LLM 情绪因子扩展（用户决策）。详见 `progress.md`。

## 这是什么

**混合架构**：传统量化 ML 模型是核心预测引擎，LLM 只负责把文本转成因子。

```
数据层（A股 行情/财务/资金流/新闻）
  ├─ 量价 / 财务 / 资金流 ──────────────────────┐
  └─ 新闻/研报/公告 →[LLM 因子]→ 情绪/事件因子 ──┤   ← Stage 2
                                                ▼
              因子层（量价因子 + LLM 因子，统一接口）
                            ▼
              模型层（LightGBM/LSTM）→ 4 类预测目标
                            ▼
                  回测 + 信号产出
```

- **核心预测引擎** = 传统量化 ML（LightGBM / LSTM），训练于「量价因子 + LLM 因子」
- **LLM 的角色** = 把新闻/研报/公告转成情绪/事件因子，作为一路因子喂给模型，**不直接出预测**
- **4 类预测目标**：① 涨跌方向（二分类）② 收益率/价格（回归）③ 选股排序（横截面 Top N）④ 买卖信号

## 分两阶段构建

- **Stage 1 — 传统量化核心** ✅：① 涨跌方向 二分类整条链做透（数据→因子→标签→模型→回测→信号），②③④ 作为复用同一套基础设施的扩展点（已留 stub）
- **Stage 2 — LLM 情绪因子扩展**（待启动）：接入文本源，LLM 抽情绪/事件因子，平滑接入因子层

## 项目结构

```
astock_quant/            主 Python 包
├── contracts.py         全局数据契约（模块间传递的 Pydantic models）
├── config/              配置（股票池、路径、回测参数、因子窗口）
├── data/                数据层（DataSource 抽象 + A股 适配器 + 缓存 + panel 数据集）
├── factors/             因子层（量价 / 财务 / 资金流 + 【LLM 因子预留】+ 注册表）
├── labels/              标签层（4 类预测目标的训练 label 生成）
├── models/              模型层（4 类目标的预测器 + 时序安全切分）
├── backtest/            回测层（逐日引擎 + A股 约束 + 绩效指标）
├── signals/             信号层（模型预测 → 可读买卖/持仓信号）
└── pipeline/            端到端编排（一条命令跑通 ①）
scripts/                 命令行入口
notebooks/               学习用 notebook（走读流程）
tests/                   测试（47 个用例，覆盖 look-ahead / T+1 / 涨跌停 / 切分纪律 / 模型 roundtrip 等）
```

完整设计（方案选型、数据契约、4 目标如何共用基础设施、LLM 因子接口预留）见 **`P1-架构设计.md`**。

## 快速上手

### 0. 系统依赖（macOS 用户必读）

LightGBM 在 macOS 需要 OpenMP runtime（`libomp`）：

```bash
brew install libomp
```

否则 `import lightgbm` 会报「dyld: Library not loaded: ... libomp.dylib」。Linux / Windows 通常自带，无需此步。

### 1. 安装 Python 依赖（用 uv 管理）

```bash
uv sync --extra dev
```

### 2. 跑 ① 涨跌方向 完整链路

最简形式（用 30 只蓝筹起步池、SETTINGS 默认日期）：

```bash
uv run python scripts/run_pipeline.py
```

带参数（换池子 / 调阈值 / 调验证集截止日）：

```bash
# 切到小池子（Stage 2 试 LLM 因子时常用）
uv run python scripts/run_pipeline.py --universe 600519,000858,000001 --quiet

# 调回测阈值 + 加大持仓上限
uv run python scripts/run_pipeline.py --buy-threshold 0.51 --sell-threshold 0.49 --max-positions 5

# 跳过回测、只评估模型（更快）
uv run python scripts/run_pipeline.py --no-backtest

# 完整参数
uv run python scripts/run_pipeline.py --help
```

### 3. 跑测试

```bash
uv run pytest tests/ -v
uv run ruff check astock_quant/ tests/
```

## 技术选型

Stage 1 **不依赖任何重框架**（无 LangGraph / langchain / backtrader）。核心依赖：`pandas / numpy / pydantic / lightgbm / scikit-learn / mootdx / akshare / stockstats / matplotlib`。

参考项目（`.p0-repos/`，作为实现参考，非依赖）：`ai-hedge-fund`（v1 回测器逻辑 + v2 架构设计）、`TradingAgents-astock`（A股 数据层 + Stage 2 LLM 封装）。选型分析见 `P0-框架选型.md`。

## 文档导航

| 文档 | 内容 |
|---|---|
| `P0-框架选型.md` | 参考框架分析与选型 |
| `P1-架构设计.md` | 架构设计：目录结构、数据契约、4 目标基础设施复用、LLM 因子接口预留 |
| `P2-数据管道.md` | 数据层：a-stock-data skill 适配 + panel 构建 + 缓存策略 |
| `P3-因子库.md` | 因子库：25 个量价/财务/资金流因子 + look-ahead 第一道防线（_winsorize 警告） |
| `P3-模型训练.md` | 模型训练：LightGBM 二分类 + 时序切分 + purge gap 第二道防线 + AUC 诚信基线 |
| `P4-回测与信号.md` | 回测引擎：逐日推进 + A股 约束 + 交易成本 + 3 组阈值实测对照 |
| `Stage1-收尾说明.md` | Stage 1 收尾 polish 7 项修复记录（reviewer + verifier 反馈） |
| `progress.md` | 实时构建进度 |
| `人话报告/` | 4 份大白话讲解（01: P0+P1 / 02: P2 / 03: P3 / 04: P4） |
| `审核/` | 9 份内部审核记录（auditor + reviewer + verifier） |
| `参考资料.md` | 用户提供的参考资料 |
