"""A股 交易约束的单元测试 —— 守住关键不变量.

按 lead 的要求，重点是：
- 「买入 T 日不能 T 日卖」（T+1）
- 「涨停日买入应被拒」（limit_up）
- 「跌停日卖出应被拒」（limit_down）
- 最小手数（< 100 拒、向下取整）
- ST 过滤

这些不变量回归一旦破，意味着回测产出的策略评估都不可信 —— 加 pytest 守住。
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from astock_quant.backtest.constraints import (
    LIMIT_PCT_DEFAULT,
    LIMIT_PCT_GROWTH,
    LIMIT_PCT_ST,
    AStockConstraints,
    check_limit_move,
    check_lot_size,
    check_st_filter,
    check_t_plus_1,
)


# ===========================================================================
# T+1
# ===========================================================================


def test_t_plus_1_blocks_same_day_sell_set():
    """买入当日的票，当日卖出必须被拒（set 形式）."""
    bought = {"600519"}
    r = check_t_plus_1("600519", date(2025, 6, 30), bought)
    assert not r.ok
    assert r.reason == "T+1"


def test_t_plus_1_blocks_same_day_sell_dict():
    """dict 形式：买入日 == today 触发拒单."""
    bought = {"600519": date(2025, 6, 30)}
    r = check_t_plus_1("600519", date(2025, 6, 30), bought)
    assert not r.ok
    assert r.reason == "T+1"


def test_t_plus_1_allows_next_day_sell():
    """次日卖出 → 允许（不在 bought_today 集合里就放行）."""
    bought = set()  # 新一天，集合清空
    r = check_t_plus_1("600519", date(2025, 7, 1), bought)
    assert r.ok


# ===========================================================================
# 涨跌停
# ===========================================================================


def test_limit_up_blocks_buy_main_board():
    """主板（600xxx）：close 触及 +10% 涨停 → 买入拒单."""
    prev_close = 100.0
    bar = pd.Series({"close": 110.0, "high": 110.0, "low": 110.0})
    r = check_limit_move("600519", bar, prev_close, "buy")
    assert not r.ok
    assert r.reason == "limit_up"


def test_limit_up_blocks_buy_growth_board():
    """创业板（300xxx）：close 触及 +20% 涨停 → 买入拒单."""
    prev_close = 100.0
    bar = pd.Series({"close": 120.0})
    r = check_limit_move("300750", bar, prev_close, "buy")
    assert not r.ok
    assert r.reason == "limit_up"


def test_limit_up_allows_non_limit_buy():
    """主板上涨 +9.5%（未到 10% 涨停）→ 买入放行."""
    prev_close = 100.0
    bar = pd.Series({"close": 109.5})
    r = check_limit_move("600519", bar, prev_close, "buy")
    assert r.ok


def test_limit_down_blocks_sell_main_board():
    """主板：close 触及 -10% 跌停 → 卖出拒单."""
    prev_close = 100.0
    bar = pd.Series({"close": 90.0})
    r = check_limit_move("600519", bar, prev_close, "sell")
    assert not r.ok
    assert r.reason == "limit_down"


def test_st_limit_pct_5pct():
    """ST 股涨跌幅 5% —— +5% 即涨停拒买."""
    prev_close = 10.0
    bar = pd.Series({"close": 10.5})
    r = check_limit_move("600519", bar, prev_close, "buy", is_st=True)
    assert not r.ok
    assert r.reason == "limit_up"


def test_limit_pct_constants_correct():
    """不变量：主板 10% / 创业 20% / ST 5%."""
    assert LIMIT_PCT_DEFAULT == 0.10
    assert LIMIT_PCT_GROWTH == 0.20
    assert LIMIT_PCT_ST == 0.05


def test_limit_move_skips_when_no_prev_close():
    """数据起点（prev_close=None）→ 无法判定，放行."""
    bar = pd.Series({"close": 110.0})
    r = check_limit_move("600519", bar, None, "buy")
    assert r.ok


# ===========================================================================
# 最小手数
# ===========================================================================


def test_lot_size_rejects_below_100_buy():
    """买入 < 100 股 → 拒单."""
    r = check_lot_size(50, "buy")
    assert not r.ok
    assert r.reason == "lot_size"


def test_lot_size_floors_to_100_multiple_buy():
    """买入 250 股 → 通过但调整到 200（向下取整到 100）."""
    r = check_lot_size(250, "buy")
    assert r.ok
    assert r.adjusted_quantity == 200


def test_lot_size_allows_arbitrary_sell_with_position():
    """卖出可以不是 100 的整数倍（A股 真实交易允许卖零头清仓）."""
    r = check_lot_size(150, "sell", current_position=150)
    assert r.ok
    assert r.adjusted_quantity == 150


# ===========================================================================
# ST 过滤
# ===========================================================================


def test_st_filter_blocks_buy_when_skip_st():
    r = check_st_filter("600519", is_st=True, skip_st=True)
    assert not r.ok
    assert r.reason == "st_filter"


def test_st_filter_allows_when_not_st():
    r = check_st_filter("600519", is_st=False, skip_st=True)
    assert r.ok


# ===========================================================================
# 组合调用 —— validate_order
# ===========================================================================


@pytest.fixture
def constraints() -> AStockConstraints:
    return AStockConstraints(skip_st=True)


def test_validate_order_t_plus_1(constraints):
    """买入当日卖出该票 → T+1 拒单."""
    bar = pd.Series({"close": 1700.0})
    r = constraints.validate_order(
        ticker="600519",
        today=date(2025, 6, 30),
        bar_today=bar,
        prev_close=1700.0,
        action="sell",
        quantity=100,
        bought_today={"600519"},
        current_position=100,
    )
    assert not r.ok
    assert r.reason == "T+1"


def test_validate_order_limit_up_buy(constraints):
    """涨停日买入 → 拒单."""
    bar = pd.Series({"close": 110.0})
    r = constraints.validate_order(
        ticker="600519",
        today=date(2025, 6, 30),
        bar_today=bar,
        prev_close=100.0,
        action="buy",
        quantity=100,
    )
    assert not r.ok
    assert r.reason == "limit_up"


def test_validate_order_happy_path(constraints):
    """正常买入 → 通过且数量取整."""
    bar = pd.Series({"close": 105.0})
    r = constraints.validate_order(
        ticker="600519",
        today=date(2025, 6, 30),
        bar_today=bar,
        prev_close=100.0,
        action="buy",
        quantity=250,
    )
    assert r.ok
    assert r.adjusted_quantity == 200
