"""P22 测试：compute_factor_frame 自动 drop 高 NaN 列。

场景：akshare 财务/资金流接口偶尔/经常全失败 → 整列 NaN → LightGBM 早停在 1 棵树
（degenerate）。在 registry 出口 drop 高 NaN 列，让训练能用剩下的弱特征训出多棵树。

测试矩阵：
- drop_nan_threshold=0.95（默认）：100% NaN 列被 drop，其它列保留
- drop_nan_threshold=1.0：只 drop 完全 NaN 的列（90% NaN 列保留）
- drop_nan_threshold=0.0：drop 所有列（极端情况，所有 NaN 比例都 >= 0）—— sanity check
- 落盘对齐：FactorFrame.factor_names 与 .data.columns 同步
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from astock_quant.factors.base import BaseFactor
from astock_quant.factors.registry import compute_factor_frame


class _ConstantFactor(BaseFactor):
    """测试用：返回一个固定 Series（按 panel 索引），可以预设 nan 比例."""

    def __init__(self, factor_name: str, nan_ratio: float = 0.0, fill_value: float = 1.0):
        self._name = factor_name
        self._nan_ratio = nan_ratio
        self._fill = fill_value

    @property
    def name(self) -> str:
        return self._name

    def compute(self, panel: pd.DataFrame, **kwargs) -> pd.Series:
        s = pd.Series(self._fill, index=panel.index, dtype=float, name=self._name)
        if self._nan_ratio > 0:
            n_nan = int(len(s) * self._nan_ratio)
            rng = np.random.default_rng(0)
            mask = rng.choice(len(s), size=n_nan, replace=False)
            s.iloc[mask] = np.nan
        return s


def _make_panel(n_dates: int = 20, n_tickers: int = 5) -> pd.DataFrame:
    """构造一个 MultiIndex=(date, ticker) 的最小行情 panel."""
    dates = pd.date_range("2025-01-01", periods=n_dates)
    tickers = [f"00000{i}" for i in range(1, n_tickers + 1)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "open": rng.uniform(10, 20, size=len(idx)),
            "high": rng.uniform(10, 20, size=len(idx)),
            "low": rng.uniform(10, 20, size=len(idx)),
            "close": rng.uniform(10, 20, size=len(idx)),
            "volume": rng.uniform(1e5, 1e6, size=len(idx)),
            "amount": rng.uniform(1e6, 1e7, size=len(idx)),
        },
        index=idx,
    )


def test_drop_100pct_nan_column():
    """默认阈值 0.95：100% NaN 列 → 被 drop；正常列 → 保留."""
    panel = _make_panel()
    factors = [
        _ConstantFactor("good_factor", nan_ratio=0.0),
        _ConstantFactor("bad_factor", nan_ratio=1.0),  # 100% NaN
    ]
    ff = compute_factor_frame(panel, factors=factors)
    assert "good_factor" in ff.factor_names
    assert "bad_factor" not in ff.factor_names
    assert list(ff.data.columns) == ff.factor_names  # 同步
    assert ff.data.shape[1] == 1


def test_threshold_boundary_98pct():
    """98% NaN 列：threshold=0.95 → drop；threshold=0.99 → 保留."""
    panel = _make_panel()
    factors = [
        _ConstantFactor("partial_factor", nan_ratio=0.98),
    ]
    # 95% 阈值：98% > 95% → drop
    ff_strict = compute_factor_frame(panel, factors=factors, drop_nan_threshold=0.95)
    assert "partial_factor" not in ff_strict.factor_names

    # 99% 阈值：98% < 99% → 保留
    ff_loose = compute_factor_frame(panel, factors=factors, drop_nan_threshold=0.99)
    assert "partial_factor" in ff_loose.factor_names


def test_threshold_1pt0_keeps_partial_nan():
    """threshold=1.0：只 drop 完全 NaN，90% NaN 列保留."""
    panel = _make_panel()
    factors = [
        _ConstantFactor("mostly_nan", nan_ratio=0.9),
        _ConstantFactor("all_nan", nan_ratio=1.0),
    ]
    ff = compute_factor_frame(panel, factors=factors, drop_nan_threshold=1.0)
    assert "mostly_nan" in ff.factor_names  # 90% < 100% → 保留
    assert "all_nan" not in ff.factor_names  # 100% >= 100% → drop


def test_drop_warning_logged(caplog):
    """drop 时应 log warning 包含被 drop 的列名 + 比例."""
    panel = _make_panel()
    factors = [
        _ConstantFactor("zzz_dead", nan_ratio=1.0),
    ]
    with caplog.at_level(logging.WARNING):
        compute_factor_frame(panel, factors=factors)
    assert any("zzz_dead" in r.message and "drop" in r.message for r in caplog.records)


def test_all_factors_dropped_returns_empty():
    """所有列都被 drop（极端情况）：返回空 FactorFrame，不 crash."""
    panel = _make_panel()
    factors = [
        _ConstantFactor("dead1", nan_ratio=1.0),
        _ConstantFactor("dead2", nan_ratio=1.0),
    ]
    ff = compute_factor_frame(panel, factors=factors)
    assert ff.factor_names == []
    assert ff.data.shape[1] == 0
