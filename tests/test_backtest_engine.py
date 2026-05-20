"""回测引擎 + Portfolio + Signal 的集成单测.

不依赖真实数据 —— 用合成 panel + 合成 Prediction，验证：
- 现金守恒（cash + holdings ≈ initial - 累计成本）
- T+1 不变量：买入当日不卖（即使 score 立刻反转）
- 涨停日买入应被拒，记入 rejection_reasons
- BacktestResult 形状正确（equity_curve / trades / positions / metrics 列齐）
- SignalGenerator 的 direction 分支 buy/hold/sell 切分正确
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from astock_quant.backtest.engine import BacktestRunConfig, BacktestEngine
from astock_quant.backtest.portfolio import Portfolio
from astock_quant.contracts import Prediction
from astock_quant.signals.generator import SignalGenerator


# ===========================================================================
# Helpers
# ===========================================================================


def _make_panel(
    tickers: list[str],
    n_days: int = 20,
    seed: int = 7,
    drift: float = 0.001,
    vol: float = 0.01,
    start: str = "2025-01-02",
) -> pd.DataFrame:
    """构造合成行情 panel."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    rows = []
    for t in tickers:
        p = 100.0
        for d in dates:
            p *= 1 + rng.normal(drift, vol)
            rows.append(
                {
                    "date": d, "ticker": t,
                    "open": p * 0.995, "high": p * 1.01, "low": p * 0.99,
                    "close": p, "volume": 1e6, "amount": p * 1e6,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


# ===========================================================================
# Portfolio 基本
# ===========================================================================


def test_portfolio_cash_conservation_buy_sell():
    """买入 + 卖出 → 现金 + 已实现盈亏 应该与「成交差额 - 各项费用」一致."""
    pf = Portfolio(initial_cash=1_000_000.0, slippage_bps=0.0, commission_rate=0.0, stamp_tax_rate=0.0)
    # 无滑点、无费用 → 完美闭合
    pf.buy("600519", 100, close=100.0, today=date(2025, 1, 2))
    assert pf.cash == 1_000_000.0 - 100 * 100.0
    pf.sell("600519", 100, close=110.0, today=date(2025, 1, 3))
    # 卖出全部 → 现金 = 初始 + 单笔盈利 1000
    assert abs(pf.cash - 1_001_000.0) < 1e-6


def test_portfolio_buy_caps_at_cash():
    """现金不足时按可用现金向下取整买入；完全不够 1 手 → 拒单."""
    pf = Portfolio(initial_cash=15_000.0, slippage_bps=0.0, commission_rate=0.0, stamp_tax_rate=0.0)
    ok, rec = pf.buy("600519", 200, close=100.0, today=date(2025, 1, 2))
    assert ok
    assert rec.quantity == 100  # 1.5w 只够 1 手 @ 100 = 1w（含手续费 0），下取整到 100

    pf2 = Portfolio(initial_cash=5_000.0, slippage_bps=0.0, commission_rate=0.0, stamp_tax_rate=0.0)
    ok2, _ = pf2.buy("600519", 100, close=100.0, today=date(2025, 1, 2))
    # 5000 不够买 100 股 @ 100 → 拒单（不足 1 手）
    assert not ok2


def test_portfolio_sell_caps_at_position():
    """卖出超过持仓量 → 按实际持仓出."""
    pf = Portfolio(initial_cash=1_000_000.0, slippage_bps=0.0, commission_rate=0.0, stamp_tax_rate=0.0)
    pf.buy("600519", 100, close=100.0, today=date(2025, 1, 2))
    ok, rec = pf.sell("600519", 500, close=110.0, today=date(2025, 1, 3))  # 想卖 500 股但只有 100
    assert ok
    assert rec.quantity == 100


def test_portfolio_total_value_marks_to_market():
    """估值 = cash + 持仓 × close."""
    pf = Portfolio(initial_cash=1_000_000.0, slippage_bps=0.0, commission_rate=0.0, stamp_tax_rate=0.0)
    pf.buy("600519", 100, close=100.0, today=date(2025, 1, 2))
    # close 涨到 120
    v = pf.total_value({"600519": 120.0})
    expected = (1_000_000.0 - 100 * 100) + 100 * 120.0
    assert abs(v - expected) < 1e-6


# ===========================================================================
# 引擎集成
# ===========================================================================


def test_engine_t_plus_1_invariant():
    """买入当日 + 立即反转的 score → 不应在当日卖出（T+1 守住）.

    构造：T0 score=0.7（强买），T0+1 score=0.1（强卖）。如果引擎错误地在 T0 收盘下单 +
    T0 再卖出，就破了 T+1。正确行为：T0 买入，T0+1 才卖出。
    """
    panel = _make_panel(["600519", "000858"], n_days=10)
    dates = panel.index.get_level_values("date").unique()
    preds = []
    # T0 强买、T1 强卖（针对 600519）；000858 一直 hold
    preds.append(
        Prediction(ticker="600519", date=dates[2].date(), target_type="direction",
                   value=1.0, score=0.70)
    )
    preds.append(
        Prediction(ticker="600519", date=dates[3].date(), target_type="direction",
                   value=0.0, score=0.10)
    )
    for d in dates[2:5]:
        preds.append(
            Prediction(ticker="000858", date=d.date(), target_type="direction",
                       value=0.0, score=0.50)
        )

    eng = BacktestEngine(
        panel,
        BacktestRunConfig(max_positions=2, buy_threshold=0.55, sell_threshold=0.45),
    )
    res = eng.run(preds)

    trades = res.trades
    # T0=dates[2] 的 buy 行 + T1=dates[3] 的 sell 行（同一 ticker），不应在同一日
    buys = trades[(trades["ticker"] == "600519") & (trades["action"] == "buy")]
    sells = trades[(trades["ticker"] == "600519") & (trades["action"] == "sell")]
    if not buys.empty and not sells.empty:
        # 每次卖出的日期必须严格 > 同 ticker 的所有买入日期里至少一个
        # 简化判定：第一笔 sell 的 date > 第一笔 buy 的 date
        assert sells.iloc[0]["date"] > buys.iloc[0]["date"], (
            f"T+1 违规：buy={buys.iloc[0]['date']}, sell={sells.iloc[0]['date']}"
        )


def test_engine_limit_up_rejection_recorded():
    """涨停日买入应被拒，并记入 rejection_reasons.

    构造：T0 close=100，T1 close=100 → T2 close=110（相对 T1 +10% 涨停）。
    prediction 放在 T2，引擎在 T2 看 prev_close=T1=100，判定 close=110 触及涨停拒单。
    （T0 是热身日，引擎从 max(panel_start, pred_start)=T2 开始推进；为让 prev_close 可见，
    panel 包含 T0/T1/T2 三日；引擎在 T2 取前一日 close 需要 prev_date 已被回测推进过 ——
    所以把 prediction 也放在 T1（让引擎走一遍 T1 后 prev_date=T1，再到 T2 时 prev_close 可读）。）
    """
    dates = pd.bdate_range("2025-01-02", periods=5)
    rows = []
    rows.append({"date": dates[0], "ticker": "600519", "open": 100, "high": 100, "low": 100, "close": 100.0, "volume": 1e6, "amount": 1e8})
    rows.append({"date": dates[1], "ticker": "600519", "open": 100, "high": 100, "low": 100, "close": 100.0, "volume": 1e6, "amount": 1e8})
    rows.append({"date": dates[2], "ticker": "600519", "open": 110, "high": 110, "low": 110, "close": 110.0, "volume": 1e6, "amount": 1e8})  # 涨停（+10%）
    rows.append({"date": dates[3], "ticker": "600519", "open": 109, "high": 112, "low": 108, "close": 111.0, "volume": 1e6, "amount": 1e8})
    rows.append({"date": dates[4], "ticker": "600519", "open": 110, "high": 113, "low": 109, "close": 112.0, "volume": 1e6, "amount": 1e8})
    panel = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    # 把 prediction 也放在 T1，让引擎从 T1 起跑、prev_date 跟上 → T2 时 prev_close 可读
    preds = [
        Prediction(ticker="600519", date=dates[1].date(),
                   target_type="direction", value=0.0, score=0.30),  # T1: 不买（score 低）
        Prediction(ticker="600519", date=dates[2].date(),
                   target_type="direction", value=1.0, score=0.80),  # T2: 强买入，但涨停应拒
    ]

    eng = BacktestEngine(panel, BacktestRunConfig(max_positions=1, buy_threshold=0.55))
    res = eng.run(preds)
    # 当日买入应被拒
    assert "limit_up" in res.metrics.get("rejection_reasons", ""), (
        f"rejection_reasons 应含 limit_up，实际：{res.metrics.get('rejection_reasons')}"
    )
    # 没有买入成交
    assert (res.trades["action"] == "buy").sum() == 0


def test_engine_returns_well_formed_result():
    """引擎产出 BacktestResult 的字段形状满足契约."""
    panel = _make_panel(["600519", "000858", "000001"], n_days=15)
    dates = panel.index.get_level_values("date").unique()
    preds = []
    rng = np.random.default_rng(0)
    for d in dates[3:]:
        for t in ["600519", "000858", "000001"]:
            preds.append(Prediction(
                ticker=t, date=d.date(), target_type="direction",
                value=1.0, score=float(rng.uniform(0.3, 0.7)),
            ))

    eng = BacktestEngine(panel, BacktestRunConfig(max_positions=2))
    res = eng.run(preds)

    # equity_curve
    assert {"portfolio_value", "cash", "holdings_value", "daily_return", "n_positions"}.issubset(
        res.equity_curve.columns
    )
    assert isinstance(res.equity_curve.index, pd.DatetimeIndex)

    # metrics 关键 key 都在
    for k in ["total_return", "sharpe", "sortino", "max_drawdown", "n_trades", "trading_days"]:
        assert k in res.metrics, f"metrics 缺 key: {k}"

    # trades 列齐
    if not res.trades.empty:
        for c in ["date", "ticker", "action", "quantity", "price", "net_cash_flow"]:
            assert c in res.trades.columns


def test_engine_empty_predictions_returns_empty_result():
    panel = _make_panel(["600519"], n_days=5)
    eng = BacktestEngine(panel, BacktestRunConfig())
    res = eng.run([])
    assert res.metrics.get("trading_days") == 0


# ===========================================================================
# missing_prediction_action（H4 修复 —— 隐性策略 → 显式 config）
# ===========================================================================


def _make_missing_pred_scenario():
    """构造场景：T1 买入 600519，T2 600519 不在 prediction 列表（数据缺失模拟），T3 又有 prediction.

    返回 (panel, predictions, dates)。
    """
    dates = pd.bdate_range("2025-01-02", periods=8)
    rows = []
    for t in ["600519", "000858"]:
        p = 100.0
        for d in dates:
            p *= 1.005
            rows.append({
                "date": d, "ticker": t,
                "open": p, "high": p * 1.01, "low": p * 0.99,
                "close": p, "volume": 1e6, "amount": 1e8,
            })
    panel = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    preds = [
        # T1：600519 强买信号、000858 持平
        Prediction(ticker="600519", date=dates[1].date(),
                   target_type="direction", value=1.0, score=0.8),
        Prediction(ticker="000858", date=dates[1].date(),
                   target_type="direction", value=0.0, score=0.5),
        # T2：600519 没 prediction（模拟 X 缺行 / 因子全 NaN 被 drop）
        Prediction(ticker="000858", date=dates[2].date(),
                   target_type="direction", value=0.0, score=0.5),
        # T3：600519 重新有 prediction（看涨）
        Prediction(ticker="600519", date=dates[3].date(),
                   target_type="direction", value=1.0, score=0.7),
        Prediction(ticker="000858", date=dates[3].date(),
                   target_type="direction", value=0.0, score=0.5),
    ]
    return panel, preds, dates


def test_engine_missing_prediction_default_is_hold():
    """关键不变量：BacktestRunConfig() 默认 missing_prediction_action == 'hold'.

    Stage 2 prep 把默认从 'liquidate' 切到 'hold'：Stage 2 LLM 因子会有大量稀疏日，
    默认 liquidate 会无谓清仓 → 默认改 hold 更稳健。如果未来回退到 liquidate 默认，
    这条测试会立刻挂。
    """
    cfg = BacktestRunConfig()
    assert cfg.missing_prediction_action == "hold", (
        f"默认 missing_prediction_action 应为 'hold'（Stage 2 prep 切换），"
        f"实际：{cfg.missing_prediction_action}"
    )


def test_engine_missing_prediction_liquidate_sells_position():
    """显式 missing_prediction_action='liquidate'：T2 600519 没 prediction → 应清仓.

    这是老行为，Stage 2 prep 切到 hold 默认后必须显式传 'liquidate' 才能复现 P4
    阶段的 14 笔交易 / +5.92% 数字。
    """
    panel, preds, _ = _make_missing_pred_scenario()
    eng = BacktestEngine(
        panel,
        BacktestRunConfig(
            max_positions=2, buy_threshold=0.55,
            missing_prediction_action="liquidate",
        ),
    )
    res = eng.run(preds)
    sells = res.trades[res.trades["action"] == "sell"]
    assert (sells["ticker"] == "600519").any(), (
        f"liquidate 模式下 T2 缺 prediction 应清仓 600519，实际 sells:\n{sells}"
    )


def test_engine_missing_prediction_hold_keeps_position():
    """默认 missing_prediction_action='hold'：T2 600519 没 prediction → 应维持持仓.

    Stage 2 prep 后这是默认行为，Stage 2 LLM 因子稀疏数据时不会被无谓清仓。
    """
    panel, preds, _ = _make_missing_pred_scenario()
    eng = BacktestEngine(
        panel,
        BacktestRunConfig(
            max_positions=2, buy_threshold=0.55,
            missing_prediction_action="hold",
        ),
    )
    res = eng.run(preds)
    sells = res.trades[
        (res.trades["action"] == "sell") & (res.trades["ticker"] == "600519")
    ]
    assert sells.empty, (
        f"hold 模式下 T2 缺 prediction 应维持 600519 持仓，但实际产生卖出:\n{sells}"
    )


def test_portfolio_avg_cost_suspended():
    """停牌日估值兜底 —— Stage 2 prep M4 修复.

    场景：T1 买入 600519 @ 100，T2 close 涨到 120 正常 mark-to-market，T3 停牌（prices 空）。
    - 老逻辑：T3 估值用 avg_cost=100 兜底 → 持仓市值 10000（假装没涨），Sharpe 被人为压低
    - 新逻辑：T3 估值用 _last_seen_close（T2 的 120）兜底 → 持仓市值 12000，更接近真实暴露

    同时验证 position_snapshot 标记 is_stale_price=True，让分析者看得见停牌污染。
    """
    from astock_quant.backtest.portfolio import Portfolio

    pf = Portfolio(
        initial_cash=1_000_000.0,
        slippage_bps=0.0, commission_rate=0.0, stamp_tax_rate=0.0,
    )
    # T1：买 100 股 @ 100，avg_cost = 100
    pf.buy("600519", 100, close=100.0, today=date(2025, 1, 2))
    assert pf.positions["600519"].avg_cost == 100.0

    # T2：close 涨到 120，正常 mark-to-market
    pf.update_last_seen_close({"600519": 120.0})
    v_t2 = pf.total_value({"600519": 120.0})
    assert abs(v_t2 - (pf.cash + 100 * 120.0)) < 1e-6

    # T3：停牌（prices 空，没给到 600519 的 close）
    v_t3 = pf.total_value({})
    # 关键不变量：估值应用上一日 close 120 兜底，而不是 avg_cost 100
    expected = pf.cash + 100 * 120.0  # = T2 持仓市值
    assert abs(v_t3 - expected) < 1e-6, (
        f"停牌日估值应用上一日 close=120 兜底（市值 {expected}），"
        f"实际 {v_t3} —— 如果是用 avg_cost=100 兜底就回归 M4 bug"
    )

    # position_snapshot 标记停牌
    snap = pf.position_snapshot({})
    assert len(snap) == 1
    assert snap[0]["close"] == 120.0  # 用 last_seen 兜底
    assert snap[0]["is_stale_price"] is True

    # 正常日：snapshot is_stale_price=False
    snap2 = pf.position_snapshot({"600519": 125.0})
    assert snap2[0]["close"] == 125.0
    assert snap2[0]["is_stale_price"] is False


def test_metrics_sortino_empty_positions():
    """全空仓（净值恒定）场景：Sharpe 和 Sortino 必须都返回 0，不能出现 -15.87 这种边界值.

    背景（auditor P4 审核观察 1）：全空仓时 daily_returns 全 0，但 excess = -daily_rf 全部为负
    → downside_dev > 0 → 老逻辑会算出 sortino ≈ -15.87，与「没承担风险」直觉相违。修复后
    以 daily_returns.std() < eps 作为单一判据，Sharpe 和 Sortino 统一兜底为 0。
    """
    import pandas as pd
    from astock_quant.backtest.metrics import compute_metrics

    dates = pd.bdate_range("2025-01-01", periods=20)
    pv = pd.Series([1_000_000.0] * 20, index=dates).rename("portfolio_value")
    ec = pd.DataFrame({"portfolio_value": pv})
    trades = pd.DataFrame(
        columns=["date", "ticker", "action", "quantity", "net_cash_flow"]
    )
    m = compute_metrics(ec, trades)
    assert m["sharpe"] == 0.0, f"Sharpe 应为 0（净值无波动），实际：{m['sharpe']}"
    assert m["sortino"] == 0.0, f"Sortino 应为 0（净值无波动），实际：{m['sortino']}"
    assert m["total_return"] == 0.0


# ===========================================================================
# Signal Generator
# ===========================================================================


def test_signal_direction_buy_sell_hold_split():
    """direction：score >= 0.55 → buy；< 0.45 → sell；中间 → hold."""
    preds = [
        Prediction(ticker="A", date=date(2025, 1, 2), target_type="direction", value=1.0, score=0.62),
        Prediction(ticker="B", date=date(2025, 1, 2), target_type="direction", value=0.0, score=0.30),
        Prediction(ticker="C", date=date(2025, 1, 2), target_type="direction", value=1.0, score=0.50),
        Prediction(ticker="D", date=date(2025, 1, 2), target_type="direction", value=0.0, score=0.55),
    ]
    rep = SignalGenerator().generate(preds)
    actions = {it.ticker: it.action for it in rep.items}
    assert actions["A"] == "buy"   # 0.62
    assert actions["B"] == "sell"  # 0.30
    assert actions["C"] == "hold"  # 0.50
    assert actions["D"] == "buy"   # 0.55 == buy_threshold


def test_signal_direction_strength_formula():
    """strength = |score - 0.5| × 2，上限 1."""
    preds = [
        Prediction(ticker="A", date=date(2025, 1, 2), target_type="direction", value=1.0, score=0.5),
        Prediction(ticker="B", date=date(2025, 1, 2), target_type="direction", value=1.0, score=0.9),
        Prediction(ticker="C", date=date(2025, 1, 2), target_type="direction", value=0.0, score=0.1),
    ]
    rep = SignalGenerator().generate(preds)
    s = {it.ticker: it.strength for it in rep.items}
    assert abs(s["A"] - 0.0) < 1e-9
    assert abs(s["B"] - 0.8) < 1e-9
    assert abs(s["C"] - 0.8) < 1e-9


def test_signal_empty_predictions():
    rep = SignalGenerator().generate([])
    assert rep.items == []


def test_signal_to_dataframe_round_trip():
    preds = [Prediction(
        ticker="600519", date=date(2025, 1, 2),
        target_type="direction", value=1.0, score=0.62,
    )]
    rep = SignalGenerator().generate(preds)
    df = rep.to_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["action"] == "buy"
    assert df.iloc[0]["ticker"] == "600519"


def test_signal_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        SignalGenerator(buy_threshold=0.4, sell_threshold=0.6)
