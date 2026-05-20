"""P11 ④ 买卖信号 —— trade_signal_label 单元测试 + 命门.

覆盖：
- 命门路径逻辑：TP only / SL only / no hit / TP before SL / SL before TP
- 边界值：close == tp_price（>=，触发）/ close == sl_price（<=，触发）
- 收盘价 only（OHLC 不介入）
- 末尾 horizon 行强制 NaN
- sl_pct >= tp_pct 抛 ValueError
- 空 panel / 缺列 / series name
- for_training 语义一致

不依赖真实缓存，全部合成 panel。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astock_quant.labels.targets import trade_signal_label


# ===========================================================================
# helpers
# ===========================================================================

def _make_panel_from_close(close_dict: dict[str, list[float]]) -> pd.DataFrame:
    """从 {ticker: [close prices]} 构造 MultiIndex(date, ticker) panel."""
    n = max(len(v) for v in close_dict.values())
    dates = pd.bdate_range("2025-01-01", periods=n)
    rows = []
    for ticker, prices in close_dict.items():
        for i, p in enumerate(prices):
            rows.append({"date": dates[i], "ticker": ticker, "close": float(p)})
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def _label_for_ticker(panel: pd.DataFrame, ticker: str) -> pd.Series:
    """取单只票的 label series（已 dropna）."""
    y = trade_signal_label(panel, horizon=3)
    return y.xs(ticker, level="ticker").dropna().sort_index()


# ===========================================================================
# 命门：路径逻辑
# ===========================================================================

def test_trade_signal_label_tp_only_hit():
    """构造连续涨价路径（每日 +10%）→ TP 被命中，标 +1."""
    # entry = 100，tp_pct=0.05 → tp_price=105
    # T+1=106, T+2=107, T+3=108 → 第 1 天就命中 TP
    closes = [100.0, 106.0, 107.0, 108.0, 109.0, 110.0]
    panel = _make_panel_from_close({"A": closes})
    y = _label_for_ticker(panel, "A")
    # T=0 (entry=100)，path[T+1]=106 >= 105 → +1
    assert y.iloc[0] == 1.0, f"应标 +1（TP），实际 {y.iloc[0]}"


def test_trade_signal_label_sl_only_hit():
    """构造连续跌价路径 → SL 被命中，标 -1."""
    # entry = 100，sl_pct=-0.03 → sl_price=97
    # T+1=96 → 第 1 天命中 SL
    closes = [100.0, 96.0, 95.0, 94.0, 93.0, 92.0]
    panel = _make_panel_from_close({"A": closes})
    y = _label_for_ticker(panel, "A")
    assert y.iloc[0] == -1.0, f"应标 -1（SL），实际 {y.iloc[0]}"


def test_trade_signal_label_no_hit_hold():
    """path 全程在 (sl_price, tp_price) 区间内 → 标 0（HOLD）."""
    # entry=100，tp=105，sl=97；path 始终 [98, 99, 100, 101] 都在区间内
    closes = [100.0, 99.0, 100.0, 101.0, 99.0, 100.0]
    panel = _make_panel_from_close({"A": closes})
    y = _label_for_ticker(panel, "A")
    # T=0: path=[99,100,101] 全在 (97,105) → HOLD=0
    assert y.iloc[0] == 0.0, f"应标 0（HOLD），实际 {y.iloc[0]}"


def test_trade_signal_label_tp_before_sl():
    """先涨 +6% 后跌 -5% → TP 先触达，标 +1."""
    # entry=100，tp=105，sl=97
    # T+1=106（TP 命中），T+2=95（SL 命中但晚了）
    closes = [100.0, 106.0, 95.0, 94.0, 93.0, 92.0]
    panel = _make_panel_from_close({"A": closes})
    y = _label_for_ticker(panel, "A")
    assert y.iloc[0] == 1.0, f"TP 先触，应标 +1，实际 {y.iloc[0]}"


def test_trade_signal_label_sl_before_tp():
    """先跌 -4% 后涨 +10% → SL 先触达，标 -1."""
    # entry=100，tp=105，sl=97
    # T+1=96（SL 命中），T+2=110（TP 命中但晚了）
    closes = [100.0, 96.0, 110.0, 111.0, 112.0, 113.0]
    panel = _make_panel_from_close({"A": closes})
    y = _label_for_ticker(panel, "A")
    assert y.iloc[0] == -1.0, f"SL 先触，应标 -1，实际 {y.iloc[0]}"


# ===========================================================================
# 边界值：>= / <= 触发语义
# ===========================================================================

def test_trade_signal_label_boundary_tp_inclusive():
    """close 恰好 = entry × (1 + tp_pct) → 触发 TP（>=，标 +1）."""
    # entry=100，tp_pct=0.05 → tp_price=105.0
    # T+1=105.0（恰好触发）
    closes = [100.0, 105.0, 104.0, 103.0, 102.0, 101.0]
    panel = _make_panel_from_close({"A": closes})
    y = trade_signal_label(panel, horizon=3, tp_pct=0.05, sl_pct=-0.03)
    y_a = y.xs("A", level="ticker").dropna()
    assert y_a.iloc[0] == 1.0, f"close == tp_price 应触发 TP (+1)，实际 {y_a.iloc[0]}"


def test_trade_signal_label_boundary_sl_inclusive():
    """close 恰好 = entry × (1 + sl_pct) → 触发 SL（<=，标 -1）."""
    # entry=100，sl_pct=-0.03 → sl_price=97.0
    # T+1=97.0（恰好触发）
    closes = [100.0, 97.0, 98.0, 99.0, 100.0, 101.0]
    panel = _make_panel_from_close({"A": closes})
    y = trade_signal_label(panel, horizon=3, tp_pct=0.05, sl_pct=-0.03)
    y_a = y.xs("A", level="ticker").dropna()
    assert y_a.iloc[0] == -1.0, f"close == sl_price 应触发 SL (-1)，实际 {y_a.iloc[0]}"


def test_trade_signal_label_just_below_tp_no_trigger():
    """close = tp_price - epsilon → 不触发 TP，继续扫后续天。"""
    # entry=100，tp=105；T+1=104.99（不触发），T+2=104.99，T+3=104.99 → HOLD
    closes = [100.0, 104.99, 104.99, 104.99, 103.0, 102.0]
    panel = _make_panel_from_close({"A": closes})
    y = trade_signal_label(panel, horizon=3, tp_pct=0.05, sl_pct=-0.03)
    y_a = y.xs("A", level="ticker").dropna()
    assert y_a.iloc[0] == 0.0, f"close < tp_price 不应触发 TP，实际 {y_a.iloc[0]}"


# ===========================================================================
# 收盘价 only（不用 OHLC）
# ===========================================================================

def test_trade_signal_label_close_price_only():
    """构造一只票 close 不触发 TP，但若用 high 列会触发 —— 标签应为 HOLD（0）.

    验证：trade_signal_label 只看 close 列，不用 OHLC 中的 high/low。
    """
    # entry=100，tp=105（close 口径）
    # close path: [104, 104, 104] 不触发 TP
    # 如果错误地用了 high 列（high=106），会错标 +1
    dates = pd.bdate_range("2025-01-01", periods=6)
    rows = []
    for i, d in enumerate(dates):
        close = [100.0, 104.0, 104.0, 104.0, 104.0, 104.0][i]
        high = [100.0, 106.0, 106.0, 106.0, 106.0, 106.0][i]  # 盘中高点超 TP，但收盘没到
        rows.append({"date": d, "ticker": "A", "close": close, "high": high})
    panel = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    y = trade_signal_label(panel, horizon=3, tp_pct=0.05, sl_pct=-0.03)
    y_a = y.xs("A", level="ticker").dropna()
    # close 没触 TP，应标 HOLD=0
    assert y_a.iloc[0] == 0.0, (
        f"trade_signal_label 不应看 high 列，close 未触 TP，应 HOLD(0)，实际 {y_a.iloc[0]}"
    )


# ===========================================================================
# 末尾 NaN
# ===========================================================================

def test_trade_signal_label_tail_horizon_nan():
    """末尾 horizon 行必须是 NaN（看不到完整未来路径）."""
    closes = [100.0 + i for i in range(20)]
    panel = _make_panel_from_close({"A": closes})
    horizon = 3
    y = trade_signal_label(panel, horizon=horizon)
    y_a = y.xs("A", level="ticker").sort_index()
    # 末尾 horizon 行必须全是 NaN
    assert y_a.iloc[-horizon:].isna().all(), (
        f"末尾 {horizon} 行应为 NaN，实际：\n{y_a.iloc[-horizon:]}"
    )
    # 中段有有效值
    assert y_a.iloc[:-(horizon + 1)].notna().any()


def test_trade_signal_label_for_training_false_same_as_true():
    """for_training=True/False 行为一致（与 direction/return/ranking 语义对齐）."""
    closes = [100.0 + i * 0.5 for i in range(20)]
    panel = _make_panel_from_close({"A": closes})
    y_tr = trade_signal_label(panel, horizon=3, for_training=True)
    y_inf = trade_signal_label(panel, horizon=3, for_training=False)
    pd.testing.assert_series_equal(y_tr, y_inf)


# ===========================================================================
# 参数校验
# ===========================================================================

def test_trade_signal_label_invalid_sl_ge_tp_raises():
    """sl_pct >= tp_pct → 抛 ValueError."""
    panel = _make_panel_from_close({"A": [100.0] * 10})
    with pytest.raises(ValueError, match="sl_pct"):
        trade_signal_label(panel, tp_pct=0.03, sl_pct=0.03)
    with pytest.raises(ValueError, match="sl_pct"):
        trade_signal_label(panel, tp_pct=0.03, sl_pct=0.05)


def test_trade_signal_label_missing_close_raises():
    """缺少 close 列 → 抛 ValueError."""
    panel = _make_panel_from_close({"A": [100.0] * 10})
    panel_no_close = panel.rename(columns={"close": "price"})
    with pytest.raises(ValueError, match="close"):
        trade_signal_label(panel_no_close)


# ===========================================================================
# 边界：空 panel
# ===========================================================================

def test_trade_signal_label_empty_panel():
    """空 panel → 返回空 Series，不抛错."""
    empty = pd.DataFrame(
        columns=["close"],
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]),
    )
    y = trade_signal_label(empty)
    assert isinstance(y, pd.Series)
    assert y.empty


# ===========================================================================
# 值域 + series name
# ===========================================================================

def test_trade_signal_label_value_domain():
    """label 值域严格 ⊂ {-1.0, 0.0, 1.0, NaN}."""
    closes = [100.0 + np.sin(i) * 5 for i in range(30)]
    panel = _make_panel_from_close({"A": closes})
    y = trade_signal_label(panel, horizon=5, tp_pct=0.03, sl_pct=-0.02)
    valid = y.dropna()
    allowed = {-1.0, 0.0, 1.0}
    unexpected = set(valid.unique()) - allowed
    assert not unexpected, f"label 出现非法值：{unexpected}"


def test_trade_signal_label_series_name():
    """输出 Series 的 name 应为 'trade_signal_label'."""
    panel = _make_panel_from_close({"A": [100.0] * 10})
    y = trade_signal_label(panel, horizon=3)
    assert y.name == "trade_signal_label"


def test_trade_signal_label_three_classes_can_appear():
    """合成足够丰富的 panel → 三种标签 (+1 / -1 / 0) 都能生成."""
    rng = np.random.default_rng(0)
    # 构造剧烈波动的 path，让三类标签都能出现
    prices = [100.0]
    for _ in range(100):
        change = rng.choice([-0.06, -0.04, 0.00, 0.04, 0.06])
        prices.append(prices[-1] * (1 + change))
    panel = _make_panel_from_close({"A": prices})
    y = trade_signal_label(panel, horizon=5, tp_pct=0.05, sl_pct=-0.03)
    valid = y.dropna()
    assert -1.0 in valid.values, "SL 类 (-1) 没有出现"
    assert 0.0 in valid.values, "HOLD 类 (0) 没有出现"
    assert 1.0 in valid.values, "TP 类 (+1) 没有出现"


# ===========================================================================
# 多 ticker 独立性
# ===========================================================================

def test_trade_signal_label_multi_ticker_independent():
    """多只票的标签独立计算，不跨股票串数据."""
    # A 全程平盘（HOLD），B 全程涨（TP）
    flat = [100.0] * 10
    rising = [100.0 * (1.06 ** i) for i in range(10)]
    panel = _make_panel_from_close({"FLAT": flat, "RISE": rising})
    y = trade_signal_label(panel, horizon=3, tp_pct=0.05, sl_pct=-0.03)
    y_flat = y.xs("FLAT", level="ticker").dropna()
    y_rise = y.xs("RISE", level="ticker").dropna()
    assert (y_flat == 0.0).all(), "平盘股应全是 HOLD(0)"
    assert (y_rise == 1.0).all(), "持续涨停股应全是 TP(+1)"
