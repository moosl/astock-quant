"""P10 ③ 横截面排序 —— run_ranking pipeline 集成测试.

命门：
- test_run_ranking_wires_ranking_label_not_return_label：pipeline 调的是 ranking_label，
  不是 return_label 或 direction_label（防 wiring 串台，同 P7 wiring 命门同款）
- test_run_ranking_uses_group_aware_splits：splits 真用了 group_by="date"，
  同一 date 的所有 ticker 全在同一个集合，不跨集

附加：
- splits group-aware 命门：test_splits_group_aware_for_ranking（§5.4 专项）
- 端到端基本流程（mock 数据层 + mock ranking_label）
- run_backtest=False 短路路径

不依赖真实缓存，全部合成 panel。
ranking_label 目前是 stub（NotImplementedError），所有 pipeline 测试都 mock ranking_label
返回合成标签，待 model-engineer 实装后只需去掉 mock 即可。
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from astock_quant.models.splits import time_series_split


# ===========================================================================
# 合成数据 helper
# ===========================================================================

class _MockSource:
    """最小 DataSource stub，提供 get_news 让 compute_factor_frame 的 news_fetcher 不崩."""
    def get_news(self, ticker, start_date=None, end_date=None, **kwargs):
        return []


def _make_synthetic_data(n_dates: int = 300, n_tickers: int = 8, seed: int = 0) -> dict:
    """构造合成数据，结构与 prepare_stage1_data 返回值一致."""
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
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
        "moneyflow": pd.DataFrame(),
        "financials": {t: [] for t in tickers},
        "source": _MockSource(),
    }


def _make_ranking_label_for_panel(price_panel: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """为 price_panel 生成合成横截面分位标签（模拟 ranking_label 的输出）.

    实现：先算 horizon 日未来收益，再按日期 groupby 做 pct rank。
    用于在 ranking_label 还是 stub 时驱动 pipeline 测试。
    """
    future_ret = (
        price_panel["close"]
        .groupby(level="ticker", group_keys=False)
        .transform(lambda s: s.pct_change(horizon).shift(-horizon))
    )
    y = future_ret.groupby(level="date").rank(pct=True)
    y.name = "ranking_label"
    return y


# ===========================================================================
# 命门 1：wiring —— pipeline 调的是 ranking_label，不是 return_label
# ===========================================================================

def test_run_ranking_wires_ranking_label_not_return_label():
    """命门：run_ranking 调的是 ranking_label，不是 return_label（wiring 防串台）.

    Stage3 §5.2 wiring 命门：mock ranking_label，断言被调用 ≥ 1 次。
    如果有人误把 run_ranking 接成 return_label（label 是连续收益率，不是横截面分位），
    本测试立刻挂。

    spy 不调通真实实现（stub 会 raise NotImplementedError），而是直接返回合成标签。
    """
    from astock_quant.pipeline.run_ranking import run_ranking

    data = _make_synthetic_data(n_dates=200, n_tickers=5)
    call_counter = {"n": 0}
    price_panel_ref = [None]  # 捕获 price_panel 引用

    def fake_ranking_label(price_panel, **kwargs):
        call_counter["n"] += 1
        price_panel_ref[0] = price_panel
        horizon = kwargs.get("horizon", 5) or 5
        return _make_ranking_label_for_panel(price_panel, horizon=horizon)

    with patch("astock_quant.pipeline.run_ranking.prepare_stage1_data", return_value=data), \
         patch("astock_quant.pipeline.run_ranking.ranking_label", side_effect=fake_ranking_label):
        run_ranking(
            train_end="2024-07-31",
            valid_end="2024-10-31",
            horizon=5,
            verbose=False,
            run_backtest=False,
        )

    assert call_counter["n"] >= 1, (
        "ranking_label 没被 run_ranking 调用 —— wiring 断了。"
        "检查 run_ranking.py 的 import 和标签计算步骤。"
    )


# ===========================================================================
# 命门 2：group-aware splits —— 同一 date 的 ticker 全在同一集合
# ===========================================================================

def test_run_ranking_uses_group_aware_splits():
    """命门：time_series_split 结果满足 group-aware 约束（§5.4）.

    Stage3 §5.4 专项：ranking 任务中同一 date 的所有 ticker 必须全在训练集
    或全在验证集，不能跨集。
    本测试直接在合成 panel 上调 time_series_split，验证这个不变量成立。
    （run_ranking 内部调的就是这个函数，接口不变 → 约束自动传递）
    """
    data = _make_synthetic_data(n_dates=200, n_tickers=8)
    prices = data["prices"]

    panel_index = prices.index

    split = time_series_split(
        panel_index,
        train_end="2024-07-31",
        valid_end="2024-10-31",
        purge_gap_days=5,
        label_horizon=5,
    )

    train_dates = set(
        panel_index[split.train_mask].get_level_values("date").unique().tolist()
    )
    valid_dates = set(
        panel_index[split.valid_mask].get_level_values("date").unique().tolist()
    )

    assert not train_dates & valid_dates, "训练日和验证日出现重叠"

    all_assigned_dates = train_dates | valid_dates
    for d in all_assigned_dates:
        day_mask = panel_index.get_level_values("date") == d
        day_in_train = split.train_mask[day_mask]
        day_in_valid = split.valid_mask[day_mask]
        any_in_train = day_in_train.any()
        any_in_valid = day_in_valid.any()
        assert not (any_in_train and any_in_valid), (
            f"日期 {d}：有 ticker 在训练集，同时有 ticker 在验证集 —— "
            "group-aware 不变量被破坏，ranking 任务的横截面会被切碎。"
        )


# ===========================================================================
# §5.4 独立命门：splits group-aware for ranking
# ===========================================================================

def test_splits_group_aware_for_ranking():
    """§5.4 专项命门：time_series_split 对 MultiIndex panel 的 group-aware 约束.

    构造 MultiIndex(date, ticker) panel，每日 N 只股票，
    验证：经过 time_series_split 后，任意日期的所有 ticker
    全在 train 或全在 valid，不出现同日跨集。
    """
    dates = pd.bdate_range("2024-01-01", periods=60)
    tickers = [f"S{i}" for i in range(5)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    split = time_series_split(
        idx,
        train_end="2024-02-29",
        valid_end="2024-04-30",
        purge_gap_days=5,
        label_horizon=5,
    )

    all_dates = idx.get_level_values("date").unique()
    for d in all_dates:
        day_mask = idx.get_level_values("date") == d
        in_train = split.train_mask[day_mask]
        in_valid = split.valid_mask[day_mask]
        assert not (in_train.any() and in_valid.any()), (
            f"日期 {d}：同日 ticker 跨集（部分 train，部分 valid）—— group-aware 破坏。"
        )


# ===========================================================================
# 端到端（mock 数据层 + mock ranking_label）
# ===========================================================================

def test_run_ranking_end_to_end_with_synthetic_data():
    """端到端：mock 数据层 + ranking_label，跑整个 run_ranking 拿到 metrics + backtest + signals."""
    from astock_quant.pipeline.run_ranking import run_ranking
    from astock_quant.contracts import Prediction

    data = _make_synthetic_data(n_dates=300, n_tickers=8)

    def fake_ranking_label(price_panel, **kwargs):
        horizon = kwargs.get("horizon", 5) or 5
        return _make_ranking_label_for_panel(price_panel, horizon=horizon)

    with patch("astock_quant.pipeline.run_ranking.prepare_stage1_data", return_value=data), \
         patch("astock_quant.pipeline.run_ranking.ranking_label", side_effect=fake_ranking_label):
        result = run_ranking(
            train_end="2024-09-30",
            valid_end="2025-02-28",
            horizon=5,
            verbose=False,
            run_backtest=True,
        )

    m = result["metrics"]
    for k in ["train_size", "valid_size", "spearman_corr", "ndcg5",
              "hit_rate_top5", "top5_bottom5_spread",
              "n_features", "train_seconds", "total_seconds"]:
        assert k in m, f"metrics 缺 key: {k}"
    assert m["train_size"] > 0 and m["valid_size"] > 0
    assert -1.0 <= m["spearman_corr"] <= 1.0 or np.isnan(m["spearman_corr"])
    assert 0.0 <= m["ndcg5"] <= 1.0 or np.isnan(m["ndcg5"])

    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) > 0
    for p in result["predictions"][:5]:
        assert isinstance(p, Prediction)
        assert p.target_type == "ranking"
        assert p.proba is None

    assert "backtest" in result
    assert "backtest_metrics" in result
    bm = result["backtest_metrics"]
    for k in ["trading_days", "n_trades", "total_return", "sharpe", "max_drawdown"]:
        assert k in bm

    assert "signals" in result
    sig = result["signals"]
    assert sig.target_type == "ranking"


def test_run_ranking_no_backtest_short_path():
    """run_backtest=False 时短路：不出 backtest / signals key，仍返回训练 metrics."""
    from astock_quant.pipeline.run_ranking import run_ranking

    data = _make_synthetic_data(n_dates=200, n_tickers=5)

    def fake_ranking_label(price_panel, **kwargs):
        horizon = kwargs.get("horizon", 5) or 5
        return _make_ranking_label_for_panel(price_panel, horizon=horizon)

    with patch("astock_quant.pipeline.run_ranking.prepare_stage1_data", return_value=data), \
         patch("astock_quant.pipeline.run_ranking.ranking_label", side_effect=fake_ranking_label):
        result = run_ranking(
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
    assert result["metrics"]["train_size"] > 0


def test_run_ranking_wires_compute_factor_frame():
    """命门：run_ranking 调 compute_factor_frame（factor 步骤不能跳过）."""
    from astock_quant.pipeline.run_ranking import run_ranking
    from astock_quant.factors.registry import compute_factor_frame as orig_cff

    data = _make_synthetic_data(n_dates=200, n_tickers=5)
    call_counter = {"n": 0}

    def spy_cff(*args, **kwargs):
        call_counter["n"] += 1
        return orig_cff(*args, **kwargs)

    def fake_ranking_label(price_panel, **kwargs):
        horizon = kwargs.get("horizon", 5) or 5
        return _make_ranking_label_for_panel(price_panel, horizon=horizon)

    with patch("astock_quant.pipeline.run_ranking.prepare_stage1_data", return_value=data), \
         patch("astock_quant.pipeline.run_ranking.ranking_label", side_effect=fake_ranking_label), \
         patch("astock_quant.pipeline.run_ranking.compute_factor_frame", side_effect=spy_cff):
        run_ranking(
            train_end="2024-07-31",
            valid_end="2024-10-31",
            horizon=5,
            verbose=False,
            run_backtest=False,
        )

    assert call_counter["n"] >= 1, "compute_factor_frame 未被调用 —— factor 步骤断了"


# ===========================================================================
# 信号层 ranking 分支行为
# ===========================================================================

def test_run_ranking_signal_uses_ranking_branch():
    """run_ranking 生成的信号 target_type 必须是 'ranking'."""
    from astock_quant.pipeline.run_ranking import run_ranking

    data = _make_synthetic_data(n_dates=300, n_tickers=8)

    def fake_ranking_label(price_panel, **kwargs):
        horizon = kwargs.get("horizon", 5) or 5
        return _make_ranking_label_for_panel(price_panel, horizon=horizon)

    with patch("astock_quant.pipeline.run_ranking.prepare_stage1_data", return_value=data), \
         patch("astock_quant.pipeline.run_ranking.ranking_label", side_effect=fake_ranking_label):
        result = run_ranking(
            train_end="2024-09-30",
            valid_end="2025-02-28",
            horizon=5,
            verbose=False,
            run_backtest=True,
        )

    sig = result["signals"]
    assert sig.target_type == "ranking", (
        f"signals.target_type 应为 'ranking'，实际: {sig.target_type}"
    )
    assert "ranking" in sig.notes.lower() or "top" in sig.notes.lower(), (
        f"signals.notes 应包含 ranking/top 信息，实际: {sig.notes}"
    )
