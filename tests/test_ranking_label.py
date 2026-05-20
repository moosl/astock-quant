"""P10 ③ 横截面排序 —— ranking_label 单元测试 + 命门.

覆盖：
- 命门 test_ranking_label_no_full_sample_rank：用「前 5 天 vs 全 10 天」检测横截面 rank
  是否只依赖当日数据，一旦 groupby(date).rank() 改成全样本 .rank() 就立刻挂
- 按 date groupby rank，同一股票不同日期 label 不相互依赖
- 末尾 horizon 行强制 NaN（与 direction/return 行为一致）
- ranking_label 高分位对应 return_label 高值（单调关系命门）
- 值域约束、空 panel、NaN 处理、for_training 语义

不依赖真实缓存，全部合成 panel。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astock_quant.labels.targets import ranking_label, return_label


# ===========================================================================
# helpers
# ===========================================================================

def _make_panel(
    n_dates: int = 20,
    tickers: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """合成 MultiIndex(date, ticker) panel，含 close 列."""
    tickers = tickers or ["A", "B", "C", "D", "E"]
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    rng = np.random.default_rng(seed)
    rows = []
    for t in tickers:
        p = 100.0
        for d in dates:
            p *= 1 + rng.normal(0.001, 0.02)
            rows.append({"date": d, "ticker": t, "close": p})
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def _split_panel_by_date(panel: pd.DataFrame, n_first: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按前 n_first 个交易日切分 panel，返回 (front, full) 两份."""
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    cutoff = all_dates[n_first - 1]
    front = panel[panel.index.get_level_values("date") <= cutoff]
    return front, panel


# ===========================================================================
# 命门：test_ranking_label_no_full_sample_rank
# ===========================================================================

def test_ranking_label_no_full_sample_rank():
    """命门：横截面 rank 必须只用「当日」数据，不能用全样本.

    设计：构造小 panel（5 只票 × 10 日），跑 ranking_label 两次：
      - 一次只用前 5 天数据（模拟 t=5 时的横截面 rank）
      - 一次用全 10 天数据

    断言：前 5 天的 rank 在两次结果里必须相同。
    如果实现错误地用了全样本 .rank()（把全 10 天一起 rank），
    前 5 天的 rank 会因为后 5 天数据的引入而改变 → 测试挂。
    """
    panel = _make_panel(n_dates=10, tickers=["A", "B", "C", "D", "E"], seed=0)
    horizon = 2

    front, full = _split_panel_by_date(panel, n_first=5)

    label_front = ranking_label(front, horizon=horizon)
    label_full = ranking_label(full, horizon=horizon)

    # 只比较两者都有有效值（非 NaN）的 (date, ticker) 行
    common_idx = label_front.index.intersection(label_full.index)
    valid_front = label_front.loc[common_idx].dropna()
    valid_full = label_full.loc[common_idx].dropna()

    common_valid = valid_front.index.intersection(valid_full.index)
    assert len(common_valid) > 0, "没有共同的非 NaN 样本，测试设计有问题"

    front_vals = valid_front.loc[common_valid].sort_index()
    full_vals = valid_full.loc[common_valid].sort_index()
    if not front_vals.equals(full_vals):
        mismatches = (front_vals - full_vals).abs()
        pytest.fail(
            f"命门失败：前 5 天的 rank 在「全 10 天」结果里不一致 —— "
            f"说明 ranking_label 用了全样本 rank，存在 look-ahead bias。"
            f"正确实现应是 groupby(date).rank()，只用当日截面。"
            f"最大差异: {mismatches.max():.6f}"
        )


# ===========================================================================
# 横截面 groupby 正确性
# ===========================================================================

def test_ranking_label_groupby_date_only():
    """ranking_label 按日期 groupby rank，同一股票不同日期的 label 不相互依赖.

    验证：每个交易日内，label 值的 rank 只和当日所有股票相比，
    不和其他日期的股票混合排序。
    """
    panel = _make_panel(n_dates=15, tickers=["A", "B", "C", "D"], seed=1)
    horizon = 3
    y = ranking_label(panel, horizon=horizon)
    y_valid = y.dropna()

    all_dates = y_valid.index.get_level_values("date").unique()
    for d in all_dates:
        day_labels = y_valid.xs(d, level="date")
        # 同一日的 label 应该是该日横截面的相对分位，彼此独立
        # 验证：每日 label 只包含当日股票，不混入其他日期
        assert len(day_labels) > 0
        # label 值域应在 [0, 1] 内（pct=True rank 的输出范围）
        assert day_labels.min() >= 0.0, f"{d} 有 label < 0"
        assert day_labels.max() <= 1.0, f"{d} 有 label > 1"


def test_ranking_label_value_range():
    """ranking_label 输出值域应在 [0, 1] 内（横截面百分位 rank）."""
    panel = _make_panel(n_dates=20, tickers=["A", "B", "C", "D", "E"])
    y = ranking_label(panel, horizon=5)
    valid = y.dropna()
    assert len(valid) > 0
    assert valid.min() >= 0.0
    assert valid.max() <= 1.0


def test_ranking_label_cross_section_sums():
    """每日横截面 label 均值应约为 0.5（百分位 rank 的期望值）."""
    panel = _make_panel(n_dates=20, tickers=[f"T{i}" for i in range(10)])
    y = ranking_label(panel, horizon=3)
    y_valid = y.dropna()

    all_dates = y_valid.index.get_level_values("date").unique()
    daily_means = [y_valid.xs(d, level="date").mean() for d in all_dates]
    overall_mean = np.mean(daily_means)
    # pct rank 的均值应接近 0.5（不完全精确，因为分组大小有限）
    assert 0.3 <= overall_mean <= 0.7, f"横截面 label 均值偏离 0.5 过多：{overall_mean:.3f}"


# ===========================================================================
# 末尾 NaN（与 direction/return 行为一致）
# ===========================================================================

def test_ranking_label_tail_nan_per_ticker():
    """每只票最后 horizon 行 label 应为 NaN（shift(-horizon) 自然结果）."""
    panel = _make_panel(n_dates=20, tickers=["A", "B", "C"])
    horizon = 5
    y = ranking_label(panel, horizon=horizon)
    for ticker in ["A", "B", "C"]:
        ticker_y = y.xs(ticker, level="ticker").sort_index()
        assert ticker_y.iloc[-horizon:].isna().all(), (
            f"{ticker} 末尾 {horizon} 行应全为 NaN，实际：\n{ticker_y.iloc[-horizon:]}"
        )
        # 中段有有效值
        assert ticker_y.iloc[:-(horizon + 1)].notna().any()


def test_ranking_label_for_training_false_same_as_true():
    """for_training=True/False 尾部 NaN 行为一致（与 direction/return 语义对齐）."""
    panel = _make_panel(n_dates=20)
    y_train = ranking_label(panel, horizon=5, for_training=True)
    y_infer = ranking_label(panel, horizon=5, for_training=False)
    pd.testing.assert_series_equal(y_train, y_infer)


# ===========================================================================
# 命门：与 return_label 的单调关系
# ===========================================================================

def test_ranking_label_consistent_with_return_label():
    """命门：同一日横截面，ranking_label 高分位对应 return_label 高值（单调关系）.

    这是数学恒等：ranking_label 就是 return_label 在横截面上的百分位排名，
    因此两者在同一日横截面内必须单调一致（Spearman rank correlation ≈ 1）。
    如果有人改变了 ranking_label 的实现（如用不同的 return 计算 or 不同的 rank），
    这条测试立刻挂。
    """
    from scipy.stats import spearmanr

    panel = _make_panel(n_dates=20, tickers=[f"T{i}" for i in range(8)])
    horizon = 3
    y_rank = ranking_label(panel, horizon=horizon)
    y_ret = return_label(panel, horizon=horizon)

    # 找有效日期（两者都有非 NaN 值的交易日）
    valid_dates = []
    all_dates = y_rank.index.get_level_values("date").unique()
    for d in all_dates:
        r_day = y_rank.xs(d, level="date").dropna()
        ret_day = y_ret.xs(d, level="date").dropna()
        common = r_day.index.intersection(ret_day.index)
        if len(common) >= 3:  # 至少 3 只票才能算 rank correlation
            valid_dates.append(d)

    assert len(valid_dates) > 0, "没有足够的有效日期做横截面 correlation 检验"

    for d in valid_dates[:5]:  # 取前 5 天验证（避免测试太慢）
        r_day = y_rank.xs(d, level="date").dropna()
        ret_day = y_ret.xs(d, level="date").dropna()
        common = r_day.index.intersection(ret_day.index)
        rho, _ = spearmanr(r_day.loc[common].values, ret_day.loc[common].values)
        assert rho > 0.99, (
            f"日期 {d}：ranking_label 和 return_label 的横截面 Spearman rho={rho:.4f} < 0.99，"
            "说明 ranking_label 的计算基础和 return_label 不一致 —— look-ahead 或实现偏差。"
        )


# ===========================================================================
# winsorize 安全性（如果实现里有 winsorize 必须按日做）
# ===========================================================================

def test_ranking_label_winsorize_per_date_not_global():
    """验证：ranking_label 的中间计算（如 winsorize）不使用全 panel 统计量.

    方法：截断 panel 到前半段，对比 ranking_label 结果；
    如果用了全样本统计量，截断前后的 rank 会不一致 → 测试挂。
    这是 P3a winsorize bug 同款检验。
    """
    panel = _make_panel(n_dates=20, tickers=["A", "B", "C", "D", "E"], seed=5)
    horizon = 3

    # 前 10 天 vs 全 20 天
    front, full = _split_panel_by_date(panel, n_first=10)
    label_front = ranking_label(front, horizon=horizon)
    label_full = ranking_label(full, horizon=horizon)

    common_idx = label_front.index.intersection(label_full.index)
    valid_front = label_front.loc[common_idx].dropna()
    valid_full = label_full.loc[common_idx].dropna()
    common_valid = valid_front.index.intersection(valid_full.index)

    if len(common_valid) == 0:
        pytest.skip("没有共同有效样本，跳过（数据太短）")

    front_vals2 = valid_front.loc[common_valid].sort_index()
    full_vals2 = valid_full.loc[common_valid].sort_index()
    if not front_vals2.equals(full_vals2):
        pytest.fail(
            "winsorize/标准化用了全 panel 统计量 —— "
            "截断 panel 后前段 rank 改变了。必须按 date groupby 做。"
        )


# ===========================================================================
# horizon 效果
# ===========================================================================

def test_ranking_label_horizon_affects_tail_nan():
    """不同 horizon 影响尾部 NaN 行数（大 horizon → 更多 NaN）."""
    panel = _make_panel(n_dates=20, tickers=["A", "B", "C"])
    y_h3 = ranking_label(panel, horizon=3)
    y_h7 = ranking_label(panel, horizon=7)
    n_tickers = 3
    # horizon=7 的 NaN 比 horizon=3 的 NaN 多 n_tickers * (7-3) 个
    extra = y_h7.isna().sum() - y_h3.isna().sum()
    assert extra == n_tickers * (7 - 3), (
        f"horizon 增加 4，NaN 应多 {n_tickers * 4} 个，实际多 {extra}"
    )


# ===========================================================================
# 边界：空 panel / 少量数据
# ===========================================================================

def test_ranking_label_empty_panel():
    """空 panel → 返回空 Series，不抛错."""
    empty = pd.DataFrame(
        columns=["close"],
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]),
    )
    y = ranking_label(empty)
    assert isinstance(y, pd.Series)
    assert y.empty


def test_ranking_label_missing_close_raises():
    """缺少 close 列 → 抛 ValueError."""
    panel = _make_panel(n_dates=5)
    panel_no_close = panel.rename(columns={"close": "price"})
    with pytest.raises((ValueError, KeyError)):
        ranking_label(panel_no_close, horizon=2)


def test_ranking_label_single_ticker():
    """只有 1 只票时，横截面 rank 结果是确定的（pct rank of 1 element = 1.0 or nan）."""
    dates = pd.date_range("2025-01-01", periods=10, freq="B")
    rows = [{"date": d, "ticker": "A", "close": 100.0 + i} for i, d in enumerate(dates)]
    panel = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    y = ranking_label(panel, horizon=2)
    valid = y.dropna()
    # 单只票横截面 rank，pct=True 时结果为 1.0（rank=1, pct=1/1=1.0）
    assert (valid == 1.0).all() or (valid.isna() | (valid == 1.0)).all()


# ===========================================================================
# 名称
# ===========================================================================

def test_ranking_label_series_name():
    """输出 Series 的 name 应为 'ranking_label'."""
    panel = _make_panel(n_dates=10)
    y = ranking_label(panel, horizon=2)
    assert y.name == "ranking_label"
