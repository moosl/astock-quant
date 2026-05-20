"""资金流因子 —— 主力 / 超大单 / 大单净流入累计.

数据来源：`data/dataset.build_moneyflow_panel()` —— MultiIndex=(date, ticker)，
columns=[main_inflow, super_inflow, large_inflow, northbound, dragon_tiger_net]。
单位：元。

────────────────────────────────────────────────────────────────────────
NaN 鲁棒性（P2 报告重要提醒）
────────────────────────────────────────────────────────────────────────
资金流端点（akshare stock_individual_fund_flow）只覆盖最近 ~6 个月，跟行情 panel
（4+ 年）对齐时会出现**大面积 NaN**。北向 / 龙虎榜在 Stage 1 没接，全 None。

设计：
1. 资金流因子接收两个 panel —— price_panel 决定输出索引（行情有的日期都要有行），
   moneyflow_panel 提供原始资金流数据。
2. 历史区间外的（资金流 panel 没覆盖的日期）一律 NaN，不做任何填充 —— 让模型 /
   下游决定怎么处理（dropna / 填 0 / 用 mask）。
3. rolling 累计在「该 ticker 有资金流数据」的窗口里算，遇到 NaN 跳过 `min_periods=1`
   保证「有几天算几天」，不要求窗口填满。

────────────────────────────────────────────────────────────────────────
为什么仍然做这一层（既然历史几乎全 NaN）
────────────────────────────────────────────────────────────────────────
- 跑日频策略时，最近 6 个月就是「实时」窗口 —— 模型对近期表现敏感的话，
  这些因子有信息量。
- P2 说「靠每日跑积累自缓存」是长期方案 —— 跑半年后历史就够长了。
- 接口形状先定下来，下游 registry / labels / models 不用因为加资金流因子重构。
"""

from __future__ import annotations

import logging

import pandas as pd

from astock_quant.factors.base import BaseFactor

logger = logging.getLogger(__name__)


# ===========================================================================
# 辅助：把资金流 panel 对齐到行情 panel 的 (date, ticker) 索引
# ===========================================================================

def _align_to_price_index(
    mf_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    col: str,
) -> pd.Series:
    """从资金流 panel 取某列，按行情 panel 的索引对齐.

    对齐方式：reindex（不做 ffill —— 资金流是「当日金额」，缺失日期不应该用前一日值）。
    缺失位置自然为 NaN。
    """
    if mf_panel is None or mf_panel.empty or col not in mf_panel.columns:
        return pd.Series(index=price_panel.index, dtype=float)
    return mf_panel[col].reindex(price_panel.index)


# ===========================================================================
# 因子基类辅助：从 kwargs 取 moneyflow panel
# ===========================================================================

class _MoneyflowBase(BaseFactor):
    """资金流因子共用：kwargs['moneyflow'] 是资金流 panel，panel 入参是行情 panel.

    输出索引始终对齐到行情 panel —— 这样所有因子（量价 + 财务 + 资金流）能在
    registry 里无缝拼成同一个 FactorFrame。
    """

    def compute(self, panel: pd.DataFrame, **kwargs) -> pd.Series:
        mf = kwargs.get("moneyflow")
        if mf is None or (isinstance(mf, pd.DataFrame) and mf.empty):
            return pd.Series(index=panel.index, dtype=float, name=self.name)
        return self._compute_inner(panel, mf)

    def _compute_inner(self, price_panel: pd.DataFrame, mf_panel: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


# ===========================================================================
# 1. 主力资金净流入累计（N 日）
# ===========================================================================

class MainInflowRolling(_MoneyflowBase):
    """主力净流入过去 N 日累计.

    用 `rolling(N, min_periods=1).sum()` —— min_periods=1 让窗口刚开始 / 资金流
    历史短的位置也能算（取已有数据求和）。NaN 仍保留为 NaN（不是 0）。
    """

    def __init__(self, window: int = 5) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"main_inflow_{self.window}d"

    def _compute_inner(self, price_panel: pd.DataFrame, mf_panel: pd.DataFrame) -> pd.Series:
        # 先在资金流 panel 上算 rolling（按 ticker 分组），再对齐到行情索引
        s_mf = mf_panel["main_inflow"].groupby(level="ticker", group_keys=False).transform(
            lambda x: x.rolling(self.window, min_periods=1).sum()
        )
        # 在 mf_panel 已计算的位置取值，缺失日期 NaN
        out = s_mf.reindex(price_panel.index)
        out.name = self.name
        return out


class SuperInflowRolling(_MoneyflowBase):
    """超大单净流入过去 N 日累计 —— 同 MainInflowRolling 逻辑."""

    def __init__(self, window: int = 5) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"super_inflow_{self.window}d"

    def _compute_inner(self, price_panel: pd.DataFrame, mf_panel: pd.DataFrame) -> pd.Series:
        s_mf = mf_panel["super_inflow"].groupby(level="ticker", group_keys=False).transform(
            lambda x: x.rolling(self.window, min_periods=1).sum()
        )
        out = s_mf.reindex(price_panel.index)
        out.name = self.name
        return out


class LargeInflowRolling(_MoneyflowBase):
    """大单净流入过去 N 日累计."""

    def __init__(self, window: int = 5) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"large_inflow_{self.window}d"

    def _compute_inner(self, price_panel: pd.DataFrame, mf_panel: pd.DataFrame) -> pd.Series:
        s_mf = mf_panel["large_inflow"].groupby(level="ticker", group_keys=False).transform(
            lambda x: x.rolling(self.window, min_periods=1).sum()
        )
        out = s_mf.reindex(price_panel.index)
        out.name = self.name
        return out


# ===========================================================================
# 2. 主力净流入 / 成交额 占比 —— 跨股票可比的强度信号
# ===========================================================================

class MainInflowToAmount(_MoneyflowBase):
    """主力净流入 / 当日成交额 —— 净流入强度（百分比）.

    单纯的主力净流入金额跨股票不可比（茅台和工行成交额差几个数量级）。
    除以当日成交额得到「主力净流入占当日成交多少比例」，∈ [-1, 1]（理论范围）。

    实际值 < 0 表示主力净卖出，> 0 净买入；绝对值越大资金动作越显眼。
    """

    @property
    def name(self) -> str:
        return "main_inflow_ratio"

    def _compute_inner(self, price_panel: pd.DataFrame, mf_panel: pd.DataFrame) -> pd.Series:
        main = mf_panel["main_inflow"].reindex(price_panel.index)
        amount = price_panel.get("amount")
        if amount is None:
            return pd.Series(index=price_panel.index, dtype=float, name=self.name)
        s = self._safe_div(main, amount)
        s = self._replace_inf(s)
        s.name = self.name
        return s


__all__ = [
    "MainInflowRolling",
    "SuperInflowRolling",
    "LargeInflowRolling",
    "MainInflowToAmount",
]
