"""A股 专属交易约束.

ai-hedge-fund v1 回测器是美股的，没有这些约束 —— 要自己加。
参考 TradingAgents-astock 对 A股 交易规则的处理。

A股 约束：
- T+1：当日买入的股票，当日不能卖出（次日才能卖）
- 涨跌停：触及涨停 / 跌停板时无法成交（涨停买不进、跌停卖不出）
- 最小手数：买入以「手」为单位，1 手 = 100 股；A股 买入数量必须是 100 的整数倍
- ST 股：可选剔除（ST 股涨跌幅限制 5%，且基本面有问题）

回测引擎下单前，先经这里过滤掉不可成交的意图。

────────────────────────────────────────────────────────────────────────
设计：单一职责
────────────────────────────────────────────────────────────────────────
本模块只做「这个意图能不能成交」的判定，不关心持仓 / 现金（那是 portfolio 的事）。
返回 CheckResult，给上层 engine 决定下一步：
- ok=True  → 接着交给 portfolio 真的下单（可能因现金不足继续受限）
- ok=False → 计为「被约束拦截」，写入 BacktestResult.metrics["n_rejected_constraint"]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from typing import Literal

import pandas as pd

# A股 涨跌停幅度：主板 / 创业板 / 科创板各异。Stage 1 简化处理：
#   - ST 股：5%
#   - 创业板（300xxx）/ 科创板（688xxx）：20%（注册制后）
#   - 其它主板：10%
# 真实回测可以更精细（看注册制实施时间、北交所 30% 等），Stage 1 这版够用且与 P0 报告口径一致。
LIMIT_PCT_DEFAULT = 0.10
LIMIT_PCT_GROWTH = 0.20  # 创业板 + 科创板
LIMIT_PCT_ST = 0.05

# 最小买入手数（1 手 = 100 股）。卖出无此限制（清仓时按实际持仓出）。
LOT_SIZE = 100


def _limit_pct_for_ticker(ticker: str, is_st: bool = False) -> float:
    """根据 ticker 前缀 + ST 状态决定涨跌停幅度.

    - ST：5%（最严，优先级最高）
    - 300xxx（创业板）/ 688xxx（科创板）：20%
    - 其余主板（6/0 开头）：10%
    """
    if is_st:
        return LIMIT_PCT_ST
    if ticker.startswith("300") or ticker.startswith("688"):
        return LIMIT_PCT_GROWTH
    return LIMIT_PCT_DEFAULT


@dataclass
class CheckResult:
    """单条下单意图的约束检查结果."""

    ok: bool
    reason: str = ""  # 拒单时的原因（"limit_up" / "limit_down" / "T+1" / "lot_size" / "st_filter"）
    adjusted_quantity: int | None = None  # 通过但需要调整数量时给出（如手数取整后）


# ---------------------------------------------------------------------------
# 单项检查 —— 引擎可单点调用，测试用
# ---------------------------------------------------------------------------


def check_limit_move(
    ticker: str,
    bar_today: pd.Series,
    prev_close: float | None,
    action: Literal["buy", "sell"],
    *,
    is_st: bool = False,
    tol: float = 1e-4,
) -> CheckResult:
    """涨跌停检查 —— 涨停买不进、跌停卖不出.

    判定口径（Stage 1 简化）：
        涨停价 = round(prev_close * (1 + limit_pct), 2)
        跌停价 = round(prev_close * (1 - limit_pct), 2)
        - bar.high == 涨停价 且 bar.low == 涨停价（一字涨停）→ 当日买入拒单
        - 更宽松一点：close >= 涨停价 - tol  → 视作涨停封板，买入拒单
        - 对称：close <= 跌停价 + tol → 视作跌停封板，卖出拒单

    放宽到「close 触及涨/跌停」而不是「一字板」，是因为：
        - 真实交易中即使盘中曾跌停后又打开，回测层难精确建模；
        - close 触及涨/跌停时，假设当日下单进不去/出不来是保守且合理的近似。

    参数：
        ticker:      6 位股票代码
        bar_today:   当日 K 线（含 close 列；high/low 可选）
        prev_close:  前一交易日收盘价；None 表示数据起点，无法判定 → 默认放行
        action:      "buy" / "sell"
        is_st:       是否 ST 股（涨跌幅 5%）
        tol:         绝对容差，对四舍五入引入的微小误差兜底
    """
    if prev_close is None or pd.isna(prev_close) or prev_close <= 0:
        return CheckResult(ok=True)

    close = bar_today.get("close")
    if close is None or pd.isna(close):
        return CheckResult(ok=False, reason="no_price")

    limit_pct = _limit_pct_for_ticker(ticker, is_st=is_st)
    limit_up = round(prev_close * (1 + limit_pct), 2)
    limit_down = round(prev_close * (1 - limit_pct), 2)

    if action == "buy" and close >= limit_up - tol:
        return CheckResult(ok=False, reason="limit_up")
    if action == "sell" and close <= limit_down + tol:
        return CheckResult(ok=False, reason="limit_down")
    return CheckResult(ok=True)


def check_t_plus_1(
    ticker: str,
    today: _date,
    bought_today: dict[str, _date] | set[str],
) -> CheckResult:
    """T+1 检查 —— 当日买入的票当日不能卖.

    `bought_today` 可以是 set[str]（只关心「今天买过哪些票」）或 dict[str, date]
    （记录每只票最后买入日，灵活点）。Stage 1 用 set 就够。
    """
    if isinstance(bought_today, set):
        if ticker in bought_today:
            return CheckResult(ok=False, reason="T+1")
        return CheckResult(ok=True)
    # dict 情况：买入日 == today 即触发
    last_buy = bought_today.get(ticker)
    if last_buy is not None and last_buy == today:
        return CheckResult(ok=False, reason="T+1")
    return CheckResult(ok=True)


def check_lot_size(quantity: int, action: Literal["buy", "sell"], *, current_position: int = 0) -> CheckResult:
    """最小手数检查 —— 买入按 100 整数倍向下取整；卖出可任意（但卖出全部 OK）.

    买入 quantity < 100 → 拒单（达不到 1 手）。
    买入 quantity >= 100 → 向下取整到百位，作为 adjusted_quantity 通过。

    卖出：A股 实际可以「不足 1 手卖出」（清仓零头），所以不强制 100 倍数。
    上层把 current_position 传进来，可以确保不卖超过持仓。
    """
    if quantity <= 0:
        return CheckResult(ok=False, reason="zero_quantity")

    if action == "buy":
        if quantity < LOT_SIZE:
            return CheckResult(ok=False, reason="lot_size")
        adjusted = (quantity // LOT_SIZE) * LOT_SIZE
        if adjusted <= 0:
            return CheckResult(ok=False, reason="lot_size")
        return CheckResult(ok=True, adjusted_quantity=adjusted)

    # action == "sell"
    sell_qty = min(quantity, current_position) if current_position > 0 else quantity
    if sell_qty <= 0:
        return CheckResult(ok=False, reason="no_position")
    return CheckResult(ok=True, adjusted_quantity=sell_qty)


def check_st_filter(ticker: str, is_st: bool, *, skip_st: bool = True) -> CheckResult:
    """ST 过滤 —— skip_st=True 时一律拒掉 ST 股的买入意图.

    Stage 1 没有 ST 名单数据源 —— 默认 is_st=False（即不过滤）。
    上层若拿到 ST 名单可显式传 is_st=True 触发拒单。
    """
    if skip_st and is_st:
        return CheckResult(ok=False, reason="st_filter")
    return CheckResult(ok=True)


# ---------------------------------------------------------------------------
# 整合 —— 引擎主流程调用
# ---------------------------------------------------------------------------


@dataclass
class AStockConstraints:
    """A股 约束集合 —— 引擎下单前调用 `validate_order` 统一过滤.

    使用：
        constraints = AStockConstraints(skip_st=True)
        result = constraints.validate_order(
            ticker="600519", today=T, bar_today=bar, prev_close=PC,
            action="buy", quantity=300, bought_today=set(), st_set=set(),
            current_position=0,
        )
        if result.ok:
            qty = result.adjusted_quantity or 300
            ...
        else:
            n_rejected[result.reason] += 1
    """

    skip_st: bool = True  # 是否过滤 ST 股
    tol: float = 1e-4  # 涨跌停价容差

    def validate_order(
        self,
        *,
        ticker: str,
        today: _date,
        bar_today: pd.Series,
        prev_close: float | None,
        action: Literal["buy", "sell"],
        quantity: int,
        bought_today: set[str] | dict[str, _date] | None = None,
        st_set: set[str] | None = None,
        current_position: int = 0,
    ) -> CheckResult:
        """完整下单意图校验 —— 任一约束失败即返回失败结果.

        校验顺序（先简单的，错误早返回）：
            ST 过滤 → 数量与手数 → T+1（仅 sell）→ 涨跌停
        """
        if bought_today is None:
            bought_today = set()
        if st_set is None:
            st_set = set()
        is_st = ticker in st_set

        # 1. ST 过滤（仅 buy；卖 ST 让它走，不强行拦）
        if action == "buy":
            r = check_st_filter(ticker, is_st=is_st, skip_st=self.skip_st)
            if not r.ok:
                return r

        # 2. 手数与数量
        r = check_lot_size(quantity, action, current_position=current_position)
        if not r.ok:
            return r
        adjusted = r.adjusted_quantity if r.adjusted_quantity is not None else quantity

        # 3. T+1（仅 sell）
        if action == "sell":
            r2 = check_t_plus_1(ticker, today, bought_today)
            if not r2.ok:
                return r2

        # 4. 涨跌停
        r3 = check_limit_move(
            ticker, bar_today, prev_close, action, is_st=is_st, tol=self.tol
        )
        if not r3.ok:
            return r3

        return CheckResult(ok=True, adjusted_quantity=adjusted)


__all__ = [
    "AStockConstraints",
    "CheckResult",
    "check_limit_move",
    "check_t_plus_1",
    "check_lot_size",
    "check_st_filter",
    "LOT_SIZE",
    "LIMIT_PCT_DEFAULT",
    "LIMIT_PCT_GROWTH",
    "LIMIT_PCT_ST",
]
