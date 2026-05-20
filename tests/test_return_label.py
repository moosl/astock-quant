"""P9 ② 收益率回归 —— return_label 单元测试.

覆盖：
- 值域是 float（不是 0/1，区别于 direction_label）
- horizon 不同时 shift 行为正确
- 末尾 horizon 行 NaN（自然 + for_training=False 语义一致）
- 命门：return_label > 0 ⇔ direction_label(threshold=0) == 1（数学恒等式）
- 起点处也 NaN（pct_change 自然结果，每只票独立）
- 多 ticker groupby 不串数据

不依赖真实缓存，全部合成 panel。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astock_quant.labels.targets import direction_label, return_label


def _make_panel(n_dates: int = 30, tickers: list[str] | None = None, seed: int = 0) -> pd.DataFrame:
    """合成 MultiIndex(date, ticker) panel，含 close 列."""
    tickers = tickers or ["A", "B", "C"]
    dates = pd.date_range("2025-01-01", periods=n_dates)
    rng = np.random.default_rng(seed)
    rows = []
    for t in tickers:
        p = 100.0
        for d in dates:
            p *= 1 + rng.normal(0.001, 0.02)
            rows.append({"date": d, "ticker": t, "close": p})
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


# ===========================================================================
# 基本语义
# ===========================================================================


def test_return_label_returns_float_series():
    """值域是 float，不是 0/1 二值（区别 direction_label 的核心点）."""
    panel = _make_panel()
    y = return_label(panel, horizon=5)
    assert isinstance(y, pd.Series)
    assert y.name == "return_label"
    # 至少有一些非 NaN 值且不全是 0/1
    valid = y.dropna()
    assert len(valid) > 0
    # float dtype
    assert valid.dtype.kind == "f"
    # 不是二值
    unique_vals = valid.unique()
    assert not set(unique_vals).issubset({0.0, 1.0}), "return_label 不应是二值"


def test_return_label_tail_nan_per_ticker():
    """每只票最后 horizon 行 label 应为 NaN（shift(-horizon) 自然结果）."""
    panel = _make_panel(n_dates=20)
    horizon = 5
    y = return_label(panel, horizon=horizon)
    for ticker in panel.index.get_level_values("ticker").unique():
        ticker_y = y.xs(ticker, level="ticker").sort_index()
        # 末尾 horizon 行必须 NaN
        assert ticker_y.iloc[-horizon:].isna().all(), (
            f"{ticker} 末尾 {horizon} 行应为 NaN，实际：\n{ticker_y.iloc[-horizon:]}"
        )
        # 中段应有有效值
        assert ticker_y.iloc[5:-horizon].notna().any()


def test_return_label_for_training_false_same_as_true_for_now():
    """for_training=True/False 当前行为一致（尾部 NaN 都来自 shift 自然结果）.

    direction_label 是同款实现，这里保持语义对齐 —— 留 for_training 参数主要是
    docstring 明示「这些样本不能算 y」。
    """
    panel = _make_panel()
    y_train = return_label(panel, horizon=5, for_training=True)
    y_infer = return_label(panel, horizon=5, for_training=False)
    pd.testing.assert_series_equal(y_train, y_infer)


def test_return_label_horizon_affects_shift():
    """不同 horizon 影响有效样本数 —— 大 horizon → 尾部 NaN 更多."""
    panel = _make_panel(n_dates=20)
    y_h5 = return_label(panel, horizon=5)
    y_h10 = return_label(panel, horizon=10)
    # h=10 有效样本比 h=5 少 N×5 个（每只票尾部多 5 行 NaN）
    n_tickers = len(panel.index.get_level_values("ticker").unique())
    assert y_h5.notna().sum() == y_h10.notna().sum() + n_tickers * 5


# ===========================================================================
# 命门：与 direction_label 的数学恒等
# ===========================================================================


def test_return_label_consistent_with_direction():
    """命门：return_label > 0 必然等价于 direction_label(threshold=0) == 1.

    这是数学恒等式 —— 二者用同样的 shift 链，direction 只是阈值化版本。
    如果哪天 return_label 的实现漂移（如改用不同的 shift / 不同 groupby），
    这条测试会立刻挂，告诉作者「你破坏了 ②①两个 label 之间的语义一致性」。
    """
    panel = _make_panel(n_dates=30)
    y_ret = return_label(panel, horizon=5)
    y_dir = direction_label(panel, horizon=5, threshold=0.0)
    # 对齐索引取共同非 NaN 样本
    common = y_ret.notna() & y_dir.notna()
    assert common.sum() > 0
    # return > 0 ⇔ direction == 1
    derived_dir = (y_ret[common] > 0).astype(float)
    pd.testing.assert_series_equal(
        derived_dir.rename("direction_label"),
        y_dir[common],
        check_names=False,
    )


def test_return_label_consistent_threshold_nonzero():
    """对任意 threshold，return > threshold ⇔ direction(threshold) == 1."""
    panel = _make_panel(n_dates=30)
    for thr in [-0.01, 0.0, 0.005, 0.02]:
        y_ret = return_label(panel, horizon=5)
        y_dir = direction_label(panel, horizon=5, threshold=thr)
        common = y_ret.notna() & y_dir.notna()
        derived_dir = (y_ret[common] > thr).astype(float)
        assert (derived_dir.values == y_dir[common].values).all(), (
            f"threshold={thr} 处 return vs direction 不一致"
        )


# ===========================================================================
# Look-ahead 边界
# ===========================================================================


def test_return_label_truncated_panel_no_future_leak():
    """模拟 panel 被 truncate_by_date 截断（只保留 T 前数据）：尾部 NaN 不会减少."""
    panel = _make_panel(n_dates=30)
    horizon = 5
    # 全量 panel 的标签
    y_full = return_label(panel, horizon=horizon)
    # 截断到第 20 日（去掉最后 10 行）
    cutoff_date = panel.index.get_level_values("date").unique()[19]
    panel_trunc = panel[panel.index.get_level_values("date") <= cutoff_date]
    y_trunc = return_label(panel_trunc, horizon=horizon)
    # 截断 panel 上，被保留的日期 ≤ cutoff，每只票末尾 horizon 行仍 NaN
    for ticker in ["A", "B", "C"]:
        tail = y_trunc.xs(ticker, level="ticker").sort_index().iloc[-horizon:]
        assert tail.isna().all()
    # 在 cutoff - horizon 之前的样本，full vs trunc 应完全一致（没有未来泄漏）
    cmp_dates = panel.index.get_level_values("date").unique()[: 20 - horizon]
    for d in cmp_dates:
        for ticker in ["A", "B", "C"]:
            if (d, ticker) in y_full.index and (d, ticker) in y_trunc.index:
                v_full = y_full.loc[(d, ticker)]
                v_trunc = y_trunc.loc[(d, ticker)]
                if not (pd.isna(v_full) or pd.isna(v_trunc)):
                    assert abs(v_full - v_trunc) < 1e-12, (
                        f"({d}, {ticker}) 截断前后 label 不一致：{v_full} vs {v_trunc}"
                    )


def test_return_label_groupby_ticker_no_cross_contamination():
    """每只票独立 groupby —— 不应跨股票串数据.

    构造一只票全 100、另一只票从 100 涨到 200：两者的 return_label 应独立计算。
    """
    dates = pd.date_range("2025-01-01", periods=15)
    rows = []
    for d in dates:
        rows.append({"date": d, "ticker": "FLAT", "close": 100.0})  # 全 100
    for i, d in enumerate(dates):
        rows.append({"date": d, "ticker": "RISE", "close": 100.0 + i * 5.0})  # 100 → 170
    panel = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    y = return_label(panel, horizon=5)
    flat_y = y.xs("FLAT", level="ticker").dropna()
    rise_y = y.xs("RISE", level="ticker").dropna()
    # FLAT 的 return 全是 0（100/100 - 1）
    assert (flat_y.values == 0.0).all()
    # RISE 的 return 全是正数
    assert (rise_y.values > 0).all()


# ===========================================================================
# 边界 / 错误处理
# ===========================================================================


def test_return_label_empty_panel():
    empty = pd.DataFrame(
        columns=["close"],
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]),
    )
    y = return_label(empty)
    assert isinstance(y, pd.Series)
    assert y.empty
    assert y.name == "return_label"


def test_return_label_missing_close_column_raises():
    dates = pd.date_range("2025-01-01", periods=5)
    panel = pd.DataFrame(
        {"open": [100, 101, 102, 103, 104]},
        index=pd.MultiIndex.from_product([dates, ["A"]], names=["date", "ticker"]),
    )
    with pytest.raises(ValueError, match="缺少 close 列"):
        return_label(panel)


def test_return_label_default_horizon_uses_settings():
    """horizon=None 时走 SETTINGS.label.horizon."""
    from astock_quant.config.settings import SETTINGS
    panel = _make_panel(n_dates=20)
    y_default = return_label(panel)
    y_explicit = return_label(panel, horizon=SETTINGS.label.horizon)
    pd.testing.assert_series_equal(y_default, y_explicit)


def test_return_label_align_xy_compatible():
    """与 align_xy 链路兼容 —— 喂入因子矩阵 + return_label 应能正常对齐."""
    from astock_quant.labels.targets import align_xy

    panel = _make_panel(n_dates=20)
    y_ret = return_label(panel, horizon=5)
    # 构造一个简单因子矩阵（与 panel 同索引）
    factors = pd.DataFrame(
        {"f0": np.arange(len(panel)), "f1": np.arange(len(panel)) * 0.1},
        index=panel.index,
    )
    X, y = align_xy(factors, y_ret)
    # align_xy 默认 drop_label_nan + drop_all_nan_rows
    # 没有 nan 因子，y 的 NaN 行会被去掉
    assert len(X) == len(y)
    assert y.notna().all()
    assert list(X.index) == list(y.index)
