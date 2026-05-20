"""绩效指标计算.

参考 ai-hedge-fund v1 src/backtesting/metrics.py + benchmarks.py，研读后重写。
基于回测产出的净值曲线，计算：
- 累计收益、年化收益
- Sharpe、Sortino 比率
- 最大回撤、回撤区间
- 胜率、盈亏比
- 对基准（沪深300）的超额收益（基准缺失时跳过）

口径（与业内 / ai-hedge-fund 对齐）：
- 年化交易日：252
- 无风险利率：默认 0.02（年）。可在 BacktestEngine 构造时覆盖。
- Sharpe = sqrt(252) × mean(excess) / std(excess)
- Sortino = sqrt(252) × mean(excess) / downside_dev
  downside_dev = sqrt(mean(min(excess, 0)^2))  —— 业内常用「下偏方差」口径
- 最大回撤：以累计净值峰值为基准的最低点比例（负数小数，如 -0.18）
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    annual_trading_days: int = 252,
    annual_rf_rate: float = 0.02,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, Any]:
    """从净值曲线 + 交易流水计算汇总指标 —— 用于 BacktestResult.metrics.

    参数：
        equity_curve:        DataFrame，index=DatetimeIndex，至少含 portfolio_value 列
        trades:              DataFrame，至少含 action / net_cash_flow / ticker 列（计胜率用）
        annual_trading_days: 年化天数，默认 252
        annual_rf_rate:      年化无风险利率，默认 2%
        benchmark_returns:   基准日收益率 Series（DatetimeIndex），缺失则跳过基准比较

    返回 dict，缺数据的指标返回 None（不报错）。
    """
    metrics: dict[str, Any] = {}

    if equity_curve is None or equity_curve.empty or "portfolio_value" not in equity_curve.columns:
        return {
            "total_return": None,
            "annualized_return": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "max_drawdown_date": None,
            "win_rate": None,
            "profit_loss_ratio": None,
            "n_trades": 0,
            "n_buy_orders": 0,
            "n_sell_orders": 0,
            "trading_days": 0,
        }

    pv = equity_curve["portfolio_value"].astype(float)
    initial = float(pv.iloc[0])
    final = float(pv.iloc[-1])
    n_days = len(pv)

    # —— 累计 / 年化收益
    if initial > 0:
        total_return = final / initial - 1.0
    else:
        total_return = float("nan")

    if n_days >= 2 and initial > 0:
        years = max(n_days / annual_trading_days, 1e-9)
        annualized_return = (final / initial) ** (1.0 / years) - 1.0
    else:
        annualized_return = None
    metrics["total_return"] = float(total_return) if not pd.isna(total_return) else None
    metrics["annualized_return"] = (
        float(annualized_return) if annualized_return is not None and not pd.isna(annualized_return) else None
    )

    # —— Sharpe / Sortino
    # 兜底口径（Sharpe / Sortino 对齐）：
    # 净值波动 std == 0（典型场景：全空仓 / 全程持仓未变）= 没承担风险 → 两者都返回 0。
    # 不要走「Sortino 公式分母用 downside 重新算」的路径 —— 全空仓时 excess = -daily_rf
    # 全部为负，downside_dev > 0，会得出 -15.87 这种「无风险但极负」的反直觉值。
    # 以「daily_returns.std() ≈ 0」作为单一判据，比 Sharpe / Sortino 分别看分母更清晰。
    daily_returns = pv.pct_change().dropna()
    if len(daily_returns) >= 2:
        daily_rf = annual_rf_rate / annual_trading_days
        excess = daily_returns - daily_rf
        mean_ex = float(excess.mean())
        ret_std = float(daily_returns.std())
        if ret_std < 1e-12:
            sharpe = 0.0
            sortino = 0.0
        else:
            sharpe = float(np.sqrt(annual_trading_days) * mean_ex / float(excess.std()))
            downside = np.minimum(excess.values, 0.0)
            dd_dev = float(np.sqrt(np.mean(downside**2)))
            if dd_dev > 1e-12:
                sortino = float(np.sqrt(annual_trading_days) * mean_ex / dd_dev)
            else:
                sortino = float("inf") if mean_ex > 0 else 0.0
        metrics["sharpe"] = sharpe
        metrics["sortino"] = sortino
    else:
        metrics["sharpe"] = None
        metrics["sortino"] = None

    # —— 最大回撤
    rolling_max = pv.cummax()
    drawdown = (pv - rolling_max) / rolling_max
    if len(drawdown) > 0 and drawdown.min() < 0:
        max_dd = float(drawdown.min())
        max_dd_date = pd.Timestamp(drawdown.idxmin()).strftime("%Y-%m-%d")
    else:
        max_dd = 0.0
        max_dd_date = None
    metrics["max_drawdown"] = max_dd
    metrics["max_drawdown_date"] = max_dd_date

    # —— 交易统计 + 胜率
    if trades is None or trades.empty:
        metrics["n_trades"] = 0
        metrics["n_buy_orders"] = 0
        metrics["n_sell_orders"] = 0
        metrics["win_rate"] = None
        metrics["profit_loss_ratio"] = None
    else:
        n_buy = int((trades["action"] == "buy").sum())
        n_sell = int((trades["action"] == "sell").sum())
        metrics["n_trades"] = int(len(trades))
        metrics["n_buy_orders"] = n_buy
        metrics["n_sell_orders"] = n_sell
        win, plr = _trade_stats(trades)
        metrics["win_rate"] = win
        metrics["profit_loss_ratio"] = plr

    metrics["trading_days"] = int(n_days)
    metrics["start_date"] = (
        pd.Timestamp(pv.index[0]).strftime("%Y-%m-%d") if n_days > 0 else None
    )
    metrics["end_date"] = (
        pd.Timestamp(pv.index[-1]).strftime("%Y-%m-%d") if n_days > 0 else None
    )

    # —— 对基准
    if benchmark_returns is not None and not benchmark_returns.empty and len(daily_returns) >= 2:
        bench_metrics = _benchmark_compare(daily_returns, benchmark_returns, annual_trading_days)
        metrics.update(bench_metrics)

    return metrics


def _trade_stats(trades: pd.DataFrame) -> tuple[float | None, float | None]:
    """胜率 + 盈亏比 —— 用「同 ticker 的 buy → sell 配对」近似估计.

    简化逻辑：对每只 ticker，按时间顺序两两配对（FIFO），sell 行的 net_cash_flow
    与对应 buy 行的 net_cash_flow 相加 = 该笔已平仓盈亏（含手续费、印花税）。

    Stage 1 简化：只看「完全平仓的票」—— 持仓未清的 ticker 不计入胜率。
    这与 ai-hedge-fund 的「以已实现 PnL 计胜率」口径一致。
    """
    if trades.empty:
        return None, None

    closed_pnls: list[float] = []
    for ticker, df in trades.groupby("ticker"):
        df = df.sort_values("date")
        buys = df[df["action"] == "buy"]
        sells = df[df["action"] == "sell"]
        if buys.empty or sells.empty:
            continue
        # FIFO 配对（quantity 加权）—— Stage 1 用「成交额加权」近似平均成本
        total_buy_cost = -float(buys["net_cash_flow"].sum())  # 现金流为负，转正
        total_buy_qty = int(buys["quantity"].sum())
        total_sell_proceeds = float(sells["net_cash_flow"].sum())  # 现金流正
        total_sell_qty = int(sells["quantity"].sum())
        if total_buy_qty == 0 or total_sell_qty == 0:
            continue
        # 已平仓部分的盈亏（按平均成本 × 平仓数量）
        avg_buy_cost = total_buy_cost / total_buy_qty
        avg_sell_price = total_sell_proceeds / total_sell_qty
        closed_qty = min(total_buy_qty, total_sell_qty)
        pnl = (avg_sell_price - avg_buy_cost) * closed_qty
        closed_pnls.append(pnl)

    if not closed_pnls:
        return None, None

    wins = [p for p in closed_pnls if p > 0]
    losses = [p for p in closed_pnls if p <= 0]
    win_rate = len(wins) / len(closed_pnls)

    if losses and wins:
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        plr = avg_win / avg_loss if avg_loss > 1e-9 else None
    else:
        plr = None
    return float(win_rate), float(plr) if plr is not None else None


def _benchmark_compare(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annual_trading_days: int,
) -> dict[str, float | None]:
    """对基准比较 —— 超额收益 / 信息比率 / β.

    把两条收益率对齐到同一组 trading 日，缺失的日子取交集。
    """
    df = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
    if len(df) < 2:
        return {
            "benchmark_total_return": None,
            "excess_return_annualized": None,
            "information_ratio": None,
            "beta": None,
        }
    s_ret = df["strategy"]
    b_ret = df["benchmark"]
    bench_total = float((1 + b_ret).prod() - 1)
    excess = s_ret - b_ret
    excess_ann = float(np.mean(excess) * annual_trading_days)
    if excess.std() > 1e-12:
        ir = float(np.sqrt(annual_trading_days) * np.mean(excess) / np.std(excess))
    else:
        ir = None
    var_b = float(np.var(b_ret))
    if var_b > 1e-12:
        cov = float(np.cov(s_ret, b_ret, ddof=0)[0, 1])
        beta = cov / var_b
    else:
        beta = None
    return {
        "benchmark_total_return": bench_total,
        "excess_return_annualized": excess_ann,
        "information_ratio": ir,
        "beta": beta,
    }


__all__ = ["compute_metrics"]
