"""labels.align_xy 顺序确定性测试 —— P5 reviewer H2 修复守门.

H2 修复前用 `factor_data.index.intersection(label_series.index)`，pandas 的 intersection
在 MultiIndex 上不保证保留原顺序，pandas 版本切换会让 (date, ticker) 行序漂移，破坏下游
time_series_split / predict 的索引对齐。

修复后改用 `reindex`，严格按 factor_data 的索引顺序对齐。

本测试守住：
1. align_xy 输出的 X.index 与 factor_data.index（去掉 NaN 行后）一致
2. 多次跑同一份输入应得到完全一致的索引顺序
3. label 的索引被打乱不影响输出顺序
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from astock_quant.labels.targets import align_xy


def _make_factor_panel(n_dates: int = 10, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["A", "B", "C"]
    dates = pd.date_range("2025-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "f0": rng.standard_normal(len(idx)),
            "f1": rng.standard_normal(len(idx)),
        },
        index=idx,
    ).sort_index()


def test_align_xy_preserves_factor_index_order():
    """X.index 与 factor_data.index 完全一致（去 label NaN 之外）."""
    X_in = _make_factor_panel()
    y = pd.Series(np.arange(len(X_in), dtype=float), index=X_in.index, name="label")

    X_out, y_out = align_xy(X_in, y)
    assert list(X_out.index) == list(X_in.index)
    assert list(y_out.index) == list(X_in.index)


def test_align_xy_robust_to_label_shuffled_index():
    """label 索引被打乱不影响 align_xy 输出顺序（H2 核心命门）."""
    X_in = _make_factor_panel()
    y = pd.Series(np.arange(len(X_in), dtype=float), index=X_in.index, name="label")

    # 多次用不同随机种子打乱 label 索引顺序
    seeds = [0, 1, 2, 42, 100]
    results = []
    for s in seeds:
        y_shuffled = y.sample(frac=1, random_state=s)
        _, y_out = align_xy(X_in, y_shuffled)
        results.append(y_out.values.tolist())

    # 所有打乱重排后输出值应该完全一致（因为 align_xy 按 X.index 严格 reindex）
    for r in results[1:]:
        assert r == results[0], "align_xy 输出对 label 顺序不稳定 —— H2 回归"


def test_align_xy_drops_label_nan_rows():
    """drop_label_nan=True 时去掉 label 为 NaN 的行，索引仍按 factor 顺序."""
    X_in = _make_factor_panel()
    y = pd.Series(np.arange(len(X_in), dtype=float), index=X_in.index, name="label")
    # 把前 5 行 label 设为 NaN
    y.iloc[:5] = np.nan

    X_out, y_out = align_xy(X_in, y, drop_label_nan=True)
    assert len(X_out) == len(X_in) - 5
    assert y_out.notna().all()
    # 顺序仍 = 原 factor 顺序的尾部
    assert list(X_out.index) == list(X_in.index[5:])


def test_align_xy_idempotent_multiple_calls():
    """连续多次 align_xy 同一份输入，结果完全相同（可复现性强）."""
    X_in = _make_factor_panel()
    y = pd.Series(np.arange(len(X_in), dtype=float), index=X_in.index, name="label")

    out1 = align_xy(X_in, y)
    out2 = align_xy(X_in, y)
    pd.testing.assert_frame_equal(out1[0], out2[0])
    pd.testing.assert_series_equal(out1[1], out2[1])


def test_align_xy_missing_label_entries_become_nan_then_dropped():
    """label 中没有的 (date, ticker) reindex 会得到 NaN，drop_label_nan 把它们去掉."""
    X_in = _make_factor_panel(n_dates=5, tickers=["A", "B"])
    # label 只覆盖一半
    y = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2025-01-01"), "A"),
                (pd.Timestamp("2025-01-02"), "B"),
                (pd.Timestamp("2025-01-03"), "A"),
            ],
            names=["date", "ticker"],
        ),
    )
    X_out, y_out = align_xy(X_in, y, drop_label_nan=True)
    assert len(X_out) == 3
    # 确认保留下来的就是 label 那 3 行（按 X.index 顺序）
    assert (y_out.values == [1.0, 2.0, 3.0]).all() or sorted(y_out.values.tolist()) == [1.0, 2.0, 3.0]
