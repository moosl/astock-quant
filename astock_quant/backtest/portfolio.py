"""持仓 + 现金 + 交易成本管理.

参考 ai-hedge-fund v1 src/backtesting/portfolio.py，研读后重写。A股 只做多，所以
比原版更简单 —— 不需要 short/cover/margin。

职责：
- 维护每只股票的持仓（股数、成本价）和账户现金
- 处理买入 / 卖出，计算交易成本（佣金双边、印花税卖出单边、滑点）
- 提供组合估值（按当日收盘价 mark-to-market）

────────────────────────────────────────────────────────────────────────
交易成本模型（与 config.settings.BacktestConfig 默认值一致；engine 用 BacktestRunConfig 时按需覆盖）
────────────────────────────────────────────────────────────────────────
- 佣金：commission_rate × 成交额（双边收取，默认 万 3 = 0.0003）
- 印花税：stamp_tax_rate × 成交额（卖出单边，默认 千 0.5 = 0.0005）
- 滑点：slippage_bps × 成交价（买入按 close × (1+滑点) 实际成交；卖出按 close × (1-滑点)）
  Stage 1 默认 5 bps（万 5）—— 对蓝筹宽松、对小票偏乐观,回测看大致区间够用。

────────────────────────────────────────────────────────────────────────
停牌估值（Stage 2 prep M4 修复）
────────────────────────────────────────────────────────────────────────
某只持仓票当日 panel 缺行（停牌 / 数据缺失），mark-to-market 需要兜底价。
- 老逻辑：用 `avg_cost` 兜底 —— 「假装没亏没赚」，掩盖真实风险（停牌当日净值平滑得像
  一池死水，Sharpe 被人为拉高,Stage 2 扩到全 A 股池后会显眼）
- 新逻辑：先用 `_last_seen_close`（上一交易日 close）兜底，再回退 `avg_cost`
  → 净值反映「最近一次见过的市场价」,更接近真实暴露

engine 主循环每日 mark-to-market 前应调 `portfolio.update_last_seen_close(today_close_map)`
更新内部字典；`holdings_value` / `position_snapshot` 在 prices 缺失某 ticker 时按这个
优先级取值。

────────────────────────────────────────────────────────────────────────
设计：纯函数式 + 显式状态
────────────────────────────────────────────────────────────────────────
不用 dict-of-dicts（ai-hedge-fund 那套，重）；A股 只做多，用「ticker → Position」
两个 dataclass 就够,访问语义清晰，类型检查友好。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """单只股票的持仓状态."""

    ticker: str
    quantity: int = 0  # 持股数（股）
    avg_cost: float = 0.0  # 持仓均价（含交易成本摊入）
    realized_pnl: float = 0.0  # 累计已实现盈亏（卖出时累加）

    def market_value(self, price: float) -> float:
        """按给定价格的市值（mark-to-market）."""
        return float(self.quantity) * float(price)

    def unrealized_pnl(self, price: float) -> float:
        """浮动盈亏（按给定价格）."""
        if self.quantity <= 0:
            return 0.0
        return (float(price) - self.avg_cost) * self.quantity


@dataclass
class TradeRecord:
    """单笔成交流水记录 —— 回测结束后落入 BacktestResult.trades."""

    date: object  # _date；用 object 避免 dataclass 与 forward-ref 冲突
    ticker: str
    action: str  # "buy" / "sell"
    quantity: int
    price: float  # 实际成交价（已含滑点）
    gross_amount: float  # 成交额（quantity × price，不含手续费）
    commission: float  # 佣金
    stamp_tax: float  # 印花税（卖出才有）
    slippage_cost: float  # 滑点成本（相对 close 的差额）
    net_cash_flow: float  # 现金流（卖入账正、买出账负，已含全部费用）
    reason: str = ""  # 触发该笔交易的原因（"signal_buy" / "rebalance" / "stop_loss" 等）


@dataclass
class Portfolio:
    """组合管理 —— 现金 + 持仓 + 交易成本.

    用法：
        pf = Portfolio(initial_cash=1_000_000)
        ok, rec = pf.buy("600519", 100, close=1700.0, today=T, reason="signal")
        v = pf.total_value(prices={"600519": 1750.0})  # mark-to-market
    """

    initial_cash: float = 1_000_000.0
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[TradeRecord] = field(default_factory=list)

    # —— 成本参数（默认与 config.settings.BacktestConfig 一致；引擎构造 BacktestRunConfig 时按需覆盖）
    commission_rate: float = 0.0003  # 佣金（双边）
    stamp_tax_rate: float = 0.0005  # 印花税（卖出单边）
    slippage_bps: float = 5.0  # 滑点（基点；1 bp = 0.0001 = 万 1）

    # M4 修复：最近一次见过的有效 close（每日由 engine 通过 update_last_seen_close 更新）。
    # 停牌 / panel 缺行时用此兜底，比 avg_cost 更接近真实市场价。
    _last_seen_close: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cash == 0.0:
            self.cash = float(self.initial_cash)

    # ------------------------------------------------------------------
    # 买入 / 卖出
    # ------------------------------------------------------------------

    def _slip(self, price: float, direction: int) -> float:
        """加滑点 —— direction=+1 买入抬价、direction=-1 卖出压价."""
        return price * (1.0 + direction * self.slippage_bps / 10_000.0)

    def buy(
        self,
        ticker: str,
        quantity: int,
        close: float,
        today: object,
        reason: str = "",
    ) -> tuple[bool, TradeRecord | None]:
        """买入 —— 现金不足时按现金可买上限向下取整到手数；成功返回 (True, record).

        失败（现金完全不够买 1 手 / quantity<=0 / 价格异常）返回 (False, None)。
        """
        if quantity <= 0 or close <= 0:
            return False, None

        exec_price = self._slip(close, +1)  # 买入加滑点
        gross = quantity * exec_price
        commission = gross * self.commission_rate
        total_cost = gross + commission

        # 现金不够 → 按手数向下取整到能买的最大
        if total_cost > self.cash:
            max_affordable_qty = int(
                self.cash / (exec_price * (1 + self.commission_rate))
            )
            max_affordable_qty = (max_affordable_qty // 100) * 100
            if max_affordable_qty <= 0:
                return False, None
            quantity = max_affordable_qty
            gross = quantity * exec_price
            commission = gross * self.commission_rate
            total_cost = gross + commission

        slippage_cost = (exec_price - close) * quantity

        # 更新持仓 —— 加权平均成本（把佣金摊入成本基价）
        pos = self.positions.get(ticker)
        if pos is None:
            pos = Position(ticker=ticker)
            self.positions[ticker] = pos
        old_qty = pos.quantity
        old_cost = pos.avg_cost
        new_qty = old_qty + quantity
        new_cost = (old_qty * old_cost + total_cost) / new_qty
        pos.quantity = new_qty
        pos.avg_cost = new_cost

        self.cash -= total_cost

        rec = TradeRecord(
            date=today,
            ticker=ticker,
            action="buy",
            quantity=quantity,
            price=exec_price,
            gross_amount=gross,
            commission=commission,
            stamp_tax=0.0,
            slippage_cost=slippage_cost,
            net_cash_flow=-total_cost,
            reason=reason,
        )
        self.trades.append(rec)
        return True, rec

    def sell(
        self,
        ticker: str,
        quantity: int,
        close: float,
        today: object,
        reason: str = "",
    ) -> tuple[bool, TradeRecord | None]:
        """卖出 —— 不足持仓时按实际持仓出（清仓零头允许，A股 真实交易也支持）.

        返回 (True, record) 表示成交；持仓为 0 / quantity<=0 时 (False, None)。
        """
        if quantity <= 0 or close <= 0:
            return False, None
        pos = self.positions.get(ticker)
        if pos is None or pos.quantity <= 0:
            return False, None
        sell_qty = min(quantity, pos.quantity)

        exec_price = self._slip(close, -1)  # 卖出压滑点
        gross = sell_qty * exec_price
        commission = gross * self.commission_rate
        stamp_tax = gross * self.stamp_tax_rate
        slippage_cost = (close - exec_price) * sell_qty
        net_proceeds = gross - commission - stamp_tax

        # 已实现盈亏（按成交价 - 持仓均价）
        realized = (exec_price - pos.avg_cost) * sell_qty - commission - stamp_tax
        pos.realized_pnl += realized
        pos.quantity -= sell_qty
        if pos.quantity == 0:
            pos.avg_cost = 0.0

        self.cash += net_proceeds

        rec = TradeRecord(
            date=today,
            ticker=ticker,
            action="sell",
            quantity=sell_qty,
            price=exec_price,
            gross_amount=gross,
            commission=commission,
            stamp_tax=stamp_tax,
            slippage_cost=slippage_cost,
            net_cash_flow=net_proceeds,
            reason=reason,
        )
        self.trades.append(rec)
        return True, rec

    # ------------------------------------------------------------------
    # 估值
    # ------------------------------------------------------------------

    def update_last_seen_close(self, prices: dict[str, float]) -> None:
        """每日更新「最近一次见过的有效 close」字典（M4 修复用）.

        engine 主循环每日 mark-to-market 前调一次：拿当日所有有效的 close（panel.loc[T] 切片）
        塞进 _last_seen_close。这样下次某 ticker 停牌 / 缺行情时，holdings_value 能用
        前一日 close 兜底，而不是用 avg_cost 假装没波动。

        只更新本调用给到的 ticker —— 缺数据的票保留上一次的值（这正是「最近一次见过」语义）。
        """
        for ticker, price in prices.items():
            if price is None or not isinstance(price, (int, float)) or price <= 0:
                continue
            self._last_seen_close[ticker] = float(price)

    def _resolve_price(self, ticker: str, prices: dict[str, float], pos: Position) -> tuple[float, bool]:
        """估值取价 —— 优先级: 当日 close > _last_seen_close > avg_cost.

        返回 (price, is_stale)：is_stale=True 表示用了 _last_seen_close 或 avg_cost 兜底
        （即当日 prices 没给到这只票的有效 close）。供 position_snapshot 标记停牌状态用。
        """
        price = prices.get(ticker)
        if price is not None and isinstance(price, (int, float)) and price > 0:
            return float(price), False
        # 当日缺行情 → 用上一次见过的 close
        last = self._last_seen_close.get(ticker)
        if last is not None and last > 0:
            return float(last), True
        # 完全没见过（极端情况：建仓当日就缺）→ 回退 avg_cost
        return float(pos.avg_cost), True

    def holdings_value(self, prices: dict[str, float]) -> float:
        """所有持仓的总市值（按给定 close 字典）.

        prices 中缺失的 ticker 优先用 _last_seen_close（上一日 close）兜底，
        都没有再回退到 avg_cost。比纯 avg_cost 兜底更接近真实暴露（M4 修复）。
        """
        total = 0.0
        for ticker, pos in self.positions.items():
            if pos.quantity <= 0:
                continue
            price, _stale = self._resolve_price(ticker, prices, pos)
            total += pos.market_value(price)
        return total

    def total_value(self, prices: dict[str, float]) -> float:
        """组合总市值 = 现金 + 持仓市值."""
        return float(self.cash) + self.holdings_value(prices)

    def n_positions(self) -> int:
        """非空持仓股数."""
        return sum(1 for p in self.positions.values() if p.quantity > 0)

    def position_snapshot(self, prices: dict[str, float]) -> list[dict]:
        """当日持仓快照 —— 用于 BacktestResult.positions DataFrame.

        返回 [{ticker, quantity, avg_cost, close, market_value, unrealized_pnl, is_stale_price}, ...]
        只包含 quantity > 0 的票。`is_stale_price=True` 表示当日 close 缺失，价格走的是
        _last_seen_close 或 avg_cost 兜底 —— 让分析者看得见停牌污染。
        """
        rows = []
        for ticker, pos in self.positions.items():
            if pos.quantity <= 0:
                continue
            price, is_stale = self._resolve_price(ticker, prices, pos)
            rows.append(
                {
                    "ticker": ticker,
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "close": price,
                    "market_value": pos.market_value(price),
                    "unrealized_pnl": pos.unrealized_pnl(price),
                    "is_stale_price": is_stale,
                }
            )
        return rows


__all__ = ["Portfolio", "Position", "TradeRecord"]
