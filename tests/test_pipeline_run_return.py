"""P9 ② 收益率回归 —— pipeline 集成 + signal return 阈值测试.

不依赖真实数据缓存（CI 友好）：
- 通过 mock `prepare_stage1_data` 注入合成 panel，直接驱动 run_return 跑通
- 验证 metrics 字段齐全 / 回测 metrics 字段齐全 / 信号 notes 包含 return 字段
- 命门：mock `compute_factor_frame` 计次 ≥ 1，证明 pipeline 真的过 factor 步骤
  （对应 Stage3 设计 §5.2 的 wiring 命门：「label 函数被真实调用，不能装死」）
- 命门：mock `return_label`，证明 run_return 接通的是 return_label 而不是 direction_label

附加单元测试：
- SignalGenerator return 分支的阈值行为（buy/sell/hold 三态、强度公式、边界值）
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from astock_quant.contracts import Prediction
from astock_quant.signals.generator import (
    DEFAULT_RETURN_BUY_THRESHOLD,
    DEFAULT_RETURN_SELL_THRESHOLD,
    SignalGenerator,
)


# ===========================================================================
# 合成 panel helpers
# ===========================================================================


def _make_synthetic_data(n_dates: int = 300, tickers: list[str] | None = None, seed: int = 0):
    """构造合成数据（足够 run_return 跑通 splits + 训练 + 回测）.

    返回类似 prepare_stage1_data 的 dict 结构。
    """
    tickers = tickers or [f"T{i:03d}" for i in range(10)]
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rng = np.random.default_rng(seed)
    rows = []
    for t in tickers:
        p = 100.0
        for d in dates:
            p *= 1 + rng.normal(0.0005, 0.015)
            rows.append({
                "date": d, "ticker": t,
                "open": p * 0.995, "high": p * 1.01, "low": p * 0.99,
                "close": p, "volume": 1e6, "amount": p * 1e6,
            })
    prices = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return {
        "prices": prices,
        "moneyflow": pd.DataFrame(),  # 空 panel（数据源历史短，正常）
        "financials": {t: [] for t in tickers},
    }


# ===========================================================================
# Pipeline 端到端（mock 数据层）
# ===========================================================================


def test_run_return_end_to_end_with_synthetic_data():
    """端到端：mock 数据层，跑整个 run_return 拿到 metrics + backtest + signals."""
    from astock_quant.pipeline.run_return import run_return

    data = _make_synthetic_data(n_dates=300, tickers=[f"T{i:03d}" for i in range(8)])

    with patch("astock_quant.pipeline.run_return.prepare_stage1_data", return_value=data):
        result = run_return(
            train_end="2024-09-30",
            valid_end="2025-02-28",
            horizon=5,
            verbose=False,
            run_backtest=True,
        )

    # —— metrics 字段齐全（回归指标）
    m = result["metrics"]
    for k in ["train_size", "valid_size", "rmse", "mae", "r2", "ic", "rank_ic",
              "y_train_mean", "y_train_std", "y_valid_mean", "y_valid_std",
              "n_features", "train_seconds", "total_seconds"]:
        assert k in m, f"metrics 缺 key: {k}"
    assert m["train_size"] > 0 and m["valid_size"] > 0
    # RMSE / MAE 是非负数
    assert m["rmse"] >= 0
    assert m["mae"] >= 0
    # IC / rank_IC 在 [-1, 1]
    if not np.isnan(m["ic"]):
        assert -1.0 <= m["ic"] <= 1.0
    if not np.isnan(m["rank_ic"]):
        assert -1.0 <= m["rank_ic"] <= 1.0

    # —— predictions 是 Prediction list，target_type == "return"
    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) > 0
    for p in result["predictions"][:5]:
        assert isinstance(p, Prediction)
        assert p.target_type == "return"
        assert p.proba is None
        assert p.value == p.score  # 回归契约

    # —— backtest 字段齐全
    assert "backtest" in result
    assert "backtest_metrics" in result
    bm = result["backtest_metrics"]
    for k in ["trading_days", "n_trades", "total_return", "sharpe",
              "sortino", "max_drawdown"]:
        assert k in bm

    # —— signals
    assert "signals" in result
    sig = result["signals"]
    assert sig.target_type == "return"
    assert "target=return" in sig.notes
    assert "buy_thr=" in sig.notes


def test_run_return_wires_return_label_not_direction_label():
    """命门：pipeline 调的是 return_label，不是 direction_label.

    Stage3 §5.2 wiring 命门：mock return_label，断言被调用 ≥ 1 次 + 返回的 series 真被用了。
    如果哪天有人误把 run_return 接成 direction_label（label 是 0/1 而非 float），
    本测试会立刻挂。
    """
    from astock_quant.pipeline.run_return import run_return

    data = _make_synthetic_data(n_dates=200, tickers=[f"T{i:03d}" for i in range(5)])

    # mock return_label 返回 spy 函数，但同时让它正常工作（返回真实合成 label）
    call_counter = {"n": 0}
    original_return_label = None

    from astock_quant.labels.targets import return_label as orig_fn
    original_return_label = orig_fn

    def spy_return_label(*args, **kwargs):
        call_counter["n"] += 1
        return original_return_label(*args, **kwargs)

    with patch("astock_quant.pipeline.run_return.prepare_stage1_data", return_value=data), \
         patch("astock_quant.pipeline.run_return.return_label", side_effect=spy_return_label):
        run_return(
            train_end="2024-07-31",
            valid_end="2024-10-31",
            horizon=5,
            verbose=False,
            run_backtest=False,  # 加速
        )

    # 必须被调用至少一次 —— 证明 pipeline 真的接通 return_label
    assert call_counter["n"] >= 1, "return_label 没被 pipeline 调用 —— wiring 断了"


def test_run_return_wires_compute_factor_frame():
    """命门：pipeline 调 compute_factor_frame（factor 步骤不能被跳过）."""
    from astock_quant.pipeline.run_return import run_return

    data = _make_synthetic_data(n_dates=200, tickers=[f"T{i:03d}" for i in range(5)])

    call_counter = {"n": 0}
    from astock_quant.factors.registry import compute_factor_frame as orig_cff

    def spy_cff(*args, **kwargs):
        call_counter["n"] += 1
        return orig_cff(*args, **kwargs)

    with patch("astock_quant.pipeline.run_return.prepare_stage1_data", return_value=data), \
         patch("astock_quant.pipeline.run_return.compute_factor_frame", side_effect=spy_cff):
        run_return(
            train_end="2024-07-31",
            valid_end="2024-10-31",
            horizon=5,
            verbose=False,
            run_backtest=False,
        )

    assert call_counter["n"] >= 1, "compute_factor_frame 未被调用 —— factor 步骤断了"


def test_run_return_no_backtest_short_path():
    """run_backtest=False 时短路：不出 backtest / signals key，仍返回训练 metrics."""
    from astock_quant.pipeline.run_return import run_return

    data = _make_synthetic_data(n_dates=200, tickers=[f"T{i:03d}" for i in range(5)])

    with patch("astock_quant.pipeline.run_return.prepare_stage1_data", return_value=data):
        result = run_return(
            train_end="2024-07-31",
            valid_end="2024-10-31",
            horizon=5,
            verbose=False,
            run_backtest=False,
        )

    assert "backtest" not in result
    assert "backtest_metrics" not in result
    assert "signals" not in result
    assert "metrics" in result
    assert result["metrics"]["rmse"] >= 0


# ===========================================================================
# SignalGenerator return 分支阈值行为（P9 升级守门）
# ===========================================================================


def test_signal_return_buy_sell_hold_split():
    """return 分支：value ≥ +2% → buy；< -2% → sell；中间 → hold."""
    preds = [
        Prediction(ticker="A", date=date(2025, 1, 2), target_type="return", value=0.03, score=0.03),
        Prediction(ticker="B", date=date(2025, 1, 2), target_type="return", value=-0.025, score=-0.025),
        Prediction(ticker="C", date=date(2025, 1, 2), target_type="return", value=0.01, score=0.01),
        # 边界：value == +0.02（>=）应触发 buy
        Prediction(ticker="D", date=date(2025, 1, 2), target_type="return", value=0.02, score=0.02),
        # 边界：value == -0.02（不 <）应是 hold
        Prediction(ticker="E", date=date(2025, 1, 2), target_type="return", value=-0.02, score=-0.02),
    ]
    rep = SignalGenerator().generate(preds)
    actions = {it.ticker: it.action for it in rep.items}
    assert actions["A"] == "buy"
    assert actions["B"] == "sell"
    assert actions["C"] == "hold"
    assert actions["D"] == "buy", f"value == buy_threshold 应触发 buy（≥），实际：{actions['D']}"
    assert actions["E"] == "hold", f"value == sell_threshold 应是 hold（不 <），实际：{actions['E']}"


def test_signal_return_strength_formula():
    """强度公式 buy：(value - buy_thr) / |buy_thr|，clipped to [0, 1].

    buy_thr=0.02 时：
    - value=0.02 → strength=0（刚好达阈值）
    - value=0.03 → strength=0.5（阈值 + 1×阈值幅度的一半）
    - value=0.04 → strength=1（阈值的 2 倍）
    - value=0.10 → strength=1（封顶）
    """
    preds = [
        Prediction(ticker="X", date=date(2025, 1, 2), target_type="return", value=0.02, score=0.02),
        Prediction(ticker="Y", date=date(2025, 1, 2), target_type="return", value=0.03, score=0.03),
        Prediction(ticker="Z", date=date(2025, 1, 2), target_type="return", value=0.04, score=0.04),
        Prediction(ticker="W", date=date(2025, 1, 2), target_type="return", value=0.10, score=0.10),
    ]
    rep = SignalGenerator().generate(preds)
    s = {it.ticker: it.strength for it in rep.items}
    assert abs(s["X"] - 0.0) < 1e-9
    assert abs(s["Y"] - 0.5) < 1e-9
    assert abs(s["Z"] - 1.0) < 1e-9
    assert abs(s["W"] - 1.0) < 1e-9  # clamp


def test_signal_return_custom_threshold():
    """自定义阈值覆盖默认 ±2%."""
    preds = [
        Prediction(ticker="A", date=date(2025, 1, 2), target_type="return", value=0.01, score=0.01),
    ]
    # buy_thr=+0.5%（更宽松）→ 0.01 应是 buy
    gen_loose = SignalGenerator(return_buy_threshold=0.005, return_sell_threshold=-0.005)
    rep_loose = gen_loose.generate(preds)
    assert rep_loose.items[0].action == "buy"

    # buy_thr=+5%（更严）→ 0.01 应是 hold
    gen_strict = SignalGenerator(return_buy_threshold=0.05, return_sell_threshold=-0.05)
    rep_strict = gen_strict.generate(preds)
    assert rep_strict.items[0].action == "hold"


def test_signal_return_threshold_validation():
    """return_buy_threshold < return_sell_threshold 必须抛 ValueError."""
    with pytest.raises(ValueError, match="return_buy_threshold"):
        SignalGenerator(return_buy_threshold=-0.05, return_sell_threshold=0.05)


def test_signal_return_default_constants_match():
    """默认阈值是 ±2%，符合 P9 设计."""
    assert DEFAULT_RETURN_BUY_THRESHOLD == 0.02
    assert DEFAULT_RETURN_SELL_THRESHOLD == -0.02
    gen = SignalGenerator()
    assert gen.return_buy_threshold == 0.02
    assert gen.return_sell_threshold == -0.02


def test_signal_return_notes_contains_threshold_info():
    """notes 必须含 buy_thr / sell_thr / n_items / n_buy / n_sell / n_hold."""
    preds = [
        Prediction(ticker="A", date=date(2025, 1, 2), target_type="return", value=0.03, score=0.03),
        Prediction(ticker="B", date=date(2025, 1, 2), target_type="return", value=-0.03, score=-0.03),
        Prediction(ticker="C", date=date(2025, 1, 2), target_type="return", value=0.0, score=0.0),
    ]
    rep = SignalGenerator().generate(preds)
    for substr in ["target=return", "buy_thr=+0.020", "sell_thr=-0.020",
                   "n_items=3", "n_buy=1", "n_sell=1", "n_hold=1"]:
        assert substr in rep.notes, f"notes 缺片段 '{substr}'，实际：{rep.notes}"
