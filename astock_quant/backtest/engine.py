"""逐日回测引擎.

逻辑重度参考 ai-hedge-fund v1 src/backtesting/engine.py + controller.py，研读后重写。
职责：
- 逐日推进（按交易日遍历回测区间）
- 每日：拿当日的模型 Prediction → 经 constraints 过滤 → 下单 → 更新 portfolio
- 记录每日净值、持仓、成交，最终产出 BacktestResult

────────────────────────────────────────────────────────────────────────
look-ahead 严控（命门）
────────────────────────────────────────────────────────────────────────
回测层是 look-ahead 的「最后一道防线」。规则：
1. 当日 T 处理流程：
   a. 用 close[T-1]（前一交易日收盘价）做涨跌停判定 —— 模拟「盘前知道昨收，盘中下单」
   b. 用 close[T]（当日收盘价）做执行价 —— 「收盘价买入 / 卖出」近似
   c. mark-to-market 用 close[T]
2. **Prediction 的 date 是 T，意味着「基于截至 T 的因子（含 T 当日）预测 T 之后的表现」**
   → 当日 T 看到 prediction[T] 后，**立刻在 T 当日按 close[T] 下单**
   这是「日终预测，下一交易日盘前下单 → 按当日 close 成交」的紧凑近似。
   严格版本应该把成交挪到 T+1（next_open），Stage 1 简化用 close[T] —— 这是常见的回测
   惯例，跟我们模型「pct_change(N).shift(-N) 用 close[T+N]/close[T]」标签口径一致。

————

资金分配（Stage 1 最简）：
- buy_threshold: score >= 0.55 视为「看涨」（高于默认阈值 0.5，留些 margin 给模型置信度）
- sell_threshold: score < 0.45 视为「看跌」（已持仓 → 清仓）
- max_positions: 单日同时持仓上限（默认 5）—— 等权分配
- 单股票仓位上限：cash 的 ~1/max_positions

4 类预测目标共用此引擎 —— 引擎只看 Prediction.score，按 target_type 决定语义：
- direction:    score = P(涨) ∈ [0,1]
- return:       score = 预期收益率（连续）
- ranking:      score = 排序分数（同日横截面比较）
- trade_signal: score = 信号强度
Stage 1 只 wire 了 direction 分支。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Any, Literal

import pandas as pd

from astock_quant.backtest.constraints import AStockConstraints
from astock_quant.backtest.metrics import compute_metrics
from astock_quant.backtest.portfolio import Portfolio
from astock_quant.contracts import BacktestResult, Prediction

logger = logging.getLogger(__name__)


@dataclass
class BacktestRunConfig:
    """回测「单次运行」的引擎参数 —— 与 `config.settings.BacktestConfig`（项目级回测配置）解耦.

    职责区分（auditor P4 审核观察 2 → P5 cleanup 修复）：
        - `config.settings.BacktestConfig`：项目级回测设置（起止日期 / 基准 / 初始资金 / 费率），
          通过 SETTINGS 全局单例使用，改它需要改配置文件
        - `BacktestRunConfig`（本类）：单次回测运行的引擎参数（含 buy/sell 阈值、max_positions、
          滑点、ST 名单等），跑测试或调超参时直接构造覆盖

    pipeline 用法：用 SETTINGS.backtest 的 3 个字段（initial_capital / 佣金 / 印花税）
    填本类的 initial_cash / commission_rate / stamp_tax_rate，其它字段走默认或显式覆盖。
    """

    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    slippage_bps: float = 5.0
    buy_threshold: float = 0.55  # score >= 0.55 才考虑买入
    sell_threshold: float = 0.45  # 持仓 + score < 0.45 → 卖出
    max_positions: int = 5  # 同时持仓上限
    skip_st: bool = True  # 过滤 ST 股
    annual_rf_rate: float = 0.02  # 年化无风险利率（指标计算）
    annual_trading_days: int = 252
    st_set: set[str] = field(default_factory=set)  # 已知 ST 名单（Stage 1 默认空）

    # P5 reviewer H4 修复 + Stage 2 prep 默认切换：当持仓的票当日不在 prediction 列表里时
    # 的策略选择。「没 prediction = 看跌」是隐性策略假设 —— 数据缺失（停牌 / 因子全 NaN 被
    # drop）会触发非预期清仓，破坏指标可信度。
    # - "hold"（默认，Stage 2 prep 切到此）：保守 ——「当日无信号 → 维持持仓」，Stage 2
    #   LLM 因子稀疏数据时更稳健，不会因为单日缺新闻就误卖
    # - "liquidate"：老行为（universe 切换 / 票被剔除 → 自动清掉持仓）。P4 阶段的 14 笔
    #   交易 / +5.92% 数字基于此模式，若想 bit-exact 复现 P4 报告需显式传该值
    missing_prediction_action: Literal["liquidate", "hold"] = "hold"


class BacktestEngine:
    """逐日回测引擎 —— 输入 Prediction 流，输出 BacktestResult.

    用法：
        engine = BacktestEngine(price_panel=panel, config=BacktestRunConfig())
        result = engine.run(predictions)
        print(result.metrics)
    """

    def __init__(
        self,
        price_panel: pd.DataFrame,
        config: BacktestRunConfig | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> None:
        """构造.

        参数：
            price_panel:        行情 panel，MultiIndex=(date, ticker)，至少含 close 列。
                                **必须** 包含回测区间的所有 (date, ticker)。引擎按
                                price_panel 的日期顺序推进。
            config:             BacktestRunConfig；缺省走默认
            benchmark_returns:  基准日收益率 Series（DatetimeIndex），缺则不算基准比较
        """
        if price_panel is None or price_panel.empty:
            raise ValueError("price_panel 为空，无法回测")
        if "close" not in price_panel.columns:
            raise ValueError(f"price_panel 必须含 close 列，当前列：{list(price_panel.columns)}")
        if not isinstance(price_panel.index, pd.MultiIndex):
            raise ValueError("price_panel 必须是 (date, ticker) MultiIndex")

        self.price_panel = price_panel.sort_index()
        self.config = config or BacktestRunConfig()
        self.benchmark_returns = benchmark_returns

        self.portfolio = Portfolio(
            initial_cash=self.config.initial_cash,
            commission_rate=self.config.commission_rate,
            stamp_tax_rate=self.config.stamp_tax_rate,
            slippage_bps=self.config.slippage_bps,
        )
        self.constraints = AStockConstraints(skip_st=self.config.skip_st)

        # —— 内部 bookkeeping
        self._bought_today: set[str] = set()  # T+1 跟踪：当日新买入的 ticker
        self._n_rejected_constraint: dict[str, int] = {}  # 按 reason 累计被拒数

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self, predictions: list[Prediction]) -> BacktestResult:
        """逐日回测.

        参数：
            predictions:  Prediction 列表。引擎按 date 分组、按 date 升序逐日处理。

        返回：BacktestResult（含 equity_curve / metrics / trades / positions）。
        """
        if not predictions:
            return self._empty_result("predictions 为空")

        # 1. 把 predictions 按 date 分组（同日多 ticker）
        pred_by_date = self._group_predictions(predictions)
        trading_dates = sorted(pred_by_date.keys())

        # 2. 用回测区间对应的 price_panel 切片做日期驱动 —— 兼容 prediction 日期与
        # panel 日期可能错位的情况（panel 是全交易日，prediction 只在 valid 日）
        # 我们以 prediction 日期为「下单日」，panel 日期为「计净值日」。
        panel_dates = self.price_panel.index.get_level_values("date").unique().sort_values()
        # 回测窗口 = max(prediction 起, panel 起) ~ min(prediction 终, panel 终)
        start = max(trading_dates[0], panel_dates.min().date())
        end = min(trading_dates[-1], panel_dates.max().date())
        bt_dates = [d for d in panel_dates if start <= d.date() <= end]
        if not bt_dates:
            return self._empty_result(f"回测区间为空：start={start}, end={end}")

        # 3. 日度净值序列
        equity_rows: list[dict[str, Any]] = []
        position_rows: list[dict[str, Any]] = []

        prev_date = None
        for ts in bt_dates:
            today: _date = ts.date()
            self._bought_today = set()  # 每日清空 T+1 标记（上一日买的票今天可卖）

            # 取当日所有 ticker 的 bar
            try:
                today_bars = self.price_panel.loc[ts]  # 一日切片，index=ticker
            except KeyError:
                # 当日无数据 → 用上一日净值续上
                if equity_rows:
                    last = equity_rows[-1]
                    equity_rows.append({**last, "date": ts})
                continue
            if isinstance(today_bars, pd.Series):
                # 当日只有一只票，loc 返回 Series；包成 DataFrame 统一处理
                today_bars = today_bars.to_frame().T
                today_bars.index = pd.Index([today_bars.index[0]], name="ticker")

            # 前一日 close（涨跌停判定）
            prev_close_map = self._prev_close_map(prev_date) if prev_date is not None else {}

            # —— 卖出阶段：先处理「持仓中 score < sell_threshold 或无 prediction（不再看涨）」的票
            self._process_sells(today, today_bars, prev_close_map, pred_by_date.get(today, []))

            # —— 买入阶段
            self._process_buys(today, today_bars, prev_close_map, pred_by_date.get(today, []))

            # —— 当日净值（按 close mark-to-market）
            close_map = self._close_map(today_bars)
            # M4 修复：每日更新「最近一次见过的有效 close」，给停牌兜底用
            self.portfolio.update_last_seen_close(close_map)
            pv = self.portfolio.total_value(close_map)
            holdings_v = self.portfolio.holdings_value(close_map)
            equity_rows.append(
                {
                    "date": ts,
                    "portfolio_value": pv,
                    "cash": self.portfolio.cash,
                    "holdings_value": holdings_v,
                    "n_positions": self.portfolio.n_positions(),
                }
            )

            # —— 当日持仓快照
            for snap in self.portfolio.position_snapshot(close_map):
                snap["date"] = ts
                position_rows.append(snap)

            prev_date = ts

        # 4. 组装 BacktestResult
        return self._assemble_result(equity_rows, position_rows)

    # ------------------------------------------------------------------
    # 内部 —— 日内动作
    # ------------------------------------------------------------------

    def _process_sells(
        self,
        today: _date,
        today_bars: pd.DataFrame,
        prev_close_map: dict[str, float],
        today_preds: list[Prediction],
    ) -> None:
        """卖出阶段 —— 已持仓的票中：
            (a) 当日 prediction score < sell_threshold → 一定清仓
            (b) 当日没出现在 prediction 名单中 → 看 config.missing_prediction_action：
                - "liquidate"（默认）：自动清仓（universe 切换 / 票被剔除时的原行为）
                - "hold"：维持持仓（数据缺失 / 因子全 NaN 时更保守，Stage 2 LLM 因子推荐）

        (P5 reviewer H4 修复：把隐性策略决策变成显式 config 选项)
        """
        held_tickers = [t for t, p in self.portfolio.positions.items() if p.quantity > 0]
        if not held_tickers:
            return

        # 当日 score 字典（仅本日有 prediction 的票）
        score_map = {p.ticker: (p.score if p.score is not None else 0.5) for p in today_preds}

        for tk in held_tickers:
            score = score_map.get(tk)
            if score is not None and score >= self.config.sell_threshold:
                continue  # score 仍看涨 → 持有
            if score is None and self.config.missing_prediction_action == "hold":
                continue  # 当日无 prediction + config 选 hold → 保守，维持持仓
            # 准备卖出
            if tk not in today_bars.index:
                continue  # 当日无数据（停牌等）→ 不卖
            bar = today_bars.loc[tk]
            close = bar.get("close")
            if close is None or pd.isna(close) or close <= 0:
                continue
            qty = self.portfolio.positions[tk].quantity
            # 约束检查
            r = self.constraints.validate_order(
                ticker=tk,
                today=today,
                bar_today=bar,
                prev_close=prev_close_map.get(tk),
                action="sell",
                quantity=qty,
                bought_today=self._bought_today,
                st_set=self.config.st_set,
                current_position=qty,
            )
            if not r.ok:
                self._n_rejected_constraint[r.reason] = (
                    self._n_rejected_constraint.get(r.reason, 0) + 1
                )
                continue
            sell_qty = r.adjusted_quantity or qty
            ok, _rec = self.portfolio.sell(
                tk, sell_qty, close=float(close), today=today,
                reason=f"score={score}" if score is not None else "not_in_universe",
            )
            if not ok:
                self._n_rejected_constraint["portfolio_reject"] = (
                    self._n_rejected_constraint.get("portfolio_reject", 0) + 1
                )

    def _process_buys(
        self,
        today: _date,
        today_bars: pd.DataFrame,
        prev_close_map: dict[str, float],
        today_preds: list[Prediction],
    ) -> None:
        """买入阶段 —— 选 score >= buy_threshold 的票，按 score 降序 Top-K 等权配资.

        K = max_positions - 当前持仓数（持仓未满才考虑新买）。
        资金分配：剩余现金 / K，每只票按这个金额向下取整到手数。
        """
        if not today_preds:
            return

        current_positions = self.portfolio.n_positions()
        slots = self.config.max_positions - current_positions
        if slots <= 0:
            return

        # 候选：score >= buy_threshold，且不在已持仓集（避免加仓 —— Stage 1 等权简化）
        held = {t for t, p in self.portfolio.positions.items() if p.quantity > 0}
        candidates = [
            p for p in today_preds
            if p.score is not None
            and p.score >= self.config.buy_threshold
            and p.ticker not in held
        ]
        if not candidates:
            return

        # 按 score 降序排，取 Top slots
        candidates.sort(key=lambda p: p.score or 0.0, reverse=True)
        top = candidates[:slots]

        # 每只票分配预算
        cash_per_slot = self.portfolio.cash / max(slots, 1)
        for pred in top:
            tk = pred.ticker
            if tk not in today_bars.index:
                continue
            bar = today_bars.loc[tk]
            close = bar.get("close")
            if close is None or pd.isna(close) or close <= 0:
                continue
            # 按预算和 close 估算手数，向下取整到 100
            est_qty = int(cash_per_slot / float(close))
            est_qty = (est_qty // 100) * 100
            if est_qty < 100:
                continue
            # 约束检查
            r = self.constraints.validate_order(
                ticker=tk,
                today=today,
                bar_today=bar,
                prev_close=prev_close_map.get(tk),
                action="buy",
                quantity=est_qty,
                bought_today=self._bought_today,
                st_set=self.config.st_set,
            )
            if not r.ok:
                self._n_rejected_constraint[r.reason] = (
                    self._n_rejected_constraint.get(r.reason, 0) + 1
                )
                continue
            buy_qty = r.adjusted_quantity or est_qty
            ok, rec = self.portfolio.buy(
                tk, buy_qty, close=float(close), today=today,
                reason=f"score={pred.score:.3f}" if pred.score is not None else "signal_buy",
            )
            if ok:
                self._bought_today.add(tk)
            else:
                self._n_rejected_constraint["portfolio_reject"] = (
                    self._n_rejected_constraint.get("portfolio_reject", 0) + 1
                )

    # ------------------------------------------------------------------
    # 内部 —— 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _group_predictions(predictions: list[Prediction]) -> dict[_date, list[Prediction]]:
        out: dict[_date, list[Prediction]] = {}
        for p in predictions:
            out.setdefault(p.date, []).append(p)
        return out

    @staticmethod
    def _close_map(today_bars: pd.DataFrame) -> dict[str, float]:
        return {
            str(tk): float(c)
            for tk, c in today_bars["close"].items()
            if c is not None and not pd.isna(c)
        }

    def _prev_close_map(self, prev_ts: pd.Timestamp) -> dict[str, float]:
        try:
            prev_bars = self.price_panel.loc[prev_ts]
        except KeyError:
            return {}
        if isinstance(prev_bars, pd.Series):
            prev_bars = prev_bars.to_frame().T
            prev_bars.index = pd.Index([prev_bars.index[0]], name="ticker")
        return self._close_map(prev_bars)

    # ------------------------------------------------------------------
    # 结果组装
    # ------------------------------------------------------------------

    def _empty_result(self, note: str) -> BacktestResult:
        logger.warning("BacktestEngine: 空结果 — %s", note)
        return BacktestResult(
            equity_curve=pd.DataFrame(
                columns=["portfolio_value", "cash", "holdings_value", "daily_return", "n_positions"]
            ),
            metrics={"note": note, "trading_days": 0},
            trades=pd.DataFrame(
                columns=[
                    "date", "ticker", "action", "quantity", "price",
                    "gross_amount", "commission", "stamp_tax",
                    "slippage_cost", "net_cash_flow", "reason",
                ]
            ),
            positions=pd.DataFrame(
                columns=["quantity", "avg_cost", "close", "market_value", "unrealized_pnl"]
            ),
        )

    def _assemble_result(
        self, equity_rows: list[dict[str, Any]], position_rows: list[dict[str, Any]]
    ) -> BacktestResult:
        # equity_curve
        ec = pd.DataFrame(equity_rows).set_index("date").sort_index()
        ec["daily_return"] = ec["portfolio_value"].pct_change()

        # trades
        if self.portfolio.trades:
            trades_df = pd.DataFrame([t.__dict__ for t in self.portfolio.trades])
        else:
            trades_df = pd.DataFrame(
                columns=[
                    "date", "ticker", "action", "quantity", "price",
                    "gross_amount", "commission", "stamp_tax",
                    "slippage_cost", "net_cash_flow", "reason",
                ]
            )

        # positions
        if position_rows:
            pos_df = pd.DataFrame(position_rows).set_index(["date", "ticker"]).sort_index()
        else:
            pos_df = pd.DataFrame(
                columns=["quantity", "avg_cost", "close", "market_value", "unrealized_pnl"]
            )

        metrics = compute_metrics(
            ec,
            trades_df,
            annual_trading_days=self.config.annual_trading_days,
            annual_rf_rate=self.config.annual_rf_rate,
            benchmark_returns=self.benchmark_returns,
        )
        # 约束拒单统计
        n_rej_total = sum(self._n_rejected_constraint.values())
        metrics["n_rejected_constraint"] = int(n_rej_total)
        # rejection_reasons 是 dict，BacktestResult.metrics schema 只装标量
        # → 序列化成 "reason1=N1;reason2=N2" 字符串
        metrics["rejection_reasons"] = (
            ";".join(f"{k}={v}" for k, v in sorted(self._n_rejected_constraint.items()))
            if self._n_rejected_constraint else ""
        )

        return BacktestResult(
            equity_curve=ec,
            metrics=metrics,
            trades=trades_df,
            positions=pos_df,
        )


__all__ = ["BacktestEngine", "BacktestRunConfig"]
