"""财务因子 —— 估值 / 盈利 / 成长.

数据来源：`data/dataset.load_financials()` → `dict[ticker, list[FinancialMetrics]]`。
财务是季度粒度、按报告期不规则发布；因子层负责把它对齐到行情 panel 的日频。

研读参考（理解后重写）：
- ai-hedge-fund v1 src/agents/fundamentals.py：ROE / 净利率 / 成长率 / PE / PB 思路

────────────────────────────────────────────────────────────────────────
关键设计：报告期 → 交易日的 forward-fill 对齐
────────────────────────────────────────────────────────────────────────
某只票交易日 T 的财务因子，用「报告期 <= T 的最近一期已发布报告」对齐 —— 即
forward-fill。astock_source.get_financials 已保证不返回未来报告期，所以「按
报告期排序后 ffill」自然防未来函数。

TODO（财报发布滞后，Stage 1 已知但未处理）：
  严格说应该用「财报实际发布日」而非「报告期末日」做截断 —— 因为年报 3/4 月才发，
  即 2024-03 实际可见的最新报告还是 2023Q3。Stage 1 简化为「报告期末日 ≤ T 即可见」，
  会让 1-3 月的财务因子轻微 forward-looking。后续可改成：
    1. astock_source.get_financials 同时返回 publish_date
    2. 此处用 publish_date <= T 做截断
  影响范围：每年 Q1（约 3 个月）的财务因子值会比真实更新；对 Stage 1 ① 涨跌方向
  二分类的损害有限（年报变动通常已 priced in），但严格的因子分析需修正。

────────────────────────────────────────────────────────────────────────
PE / PB 的特殊处理
────────────────────────────────────────────────────────────────────────
astock_source 只把 PE / PB / 总股本 / 流通股本挂在「最新一期」报告上（来自
腾讯实时快照），历史报告期都是 None —— 这是为了避免误用历史估值快照。所以
PE / PB 因子在历史交易日的值会大量 NaN，只有「靠近最新报告期」的近期才有值。
对 Stage 1 训练影响：训练集主要在历史区间，PE / PB 几乎全 NaN，模型基本学不到；
要用估值因子需要先补齐 PE / PB 的历史数据（Stage 1 暂留 TODO）。
"""

from __future__ import annotations

import logging

import pandas as pd

from astock_quant.contracts import FinancialMetrics
from astock_quant.factors.base import BaseFactor

logger = logging.getLogger(__name__)


# ===========================================================================
# 工具：把 dict[ticker, list[FinancialMetrics]] 对齐到行情 panel 的 (date, ticker)
# ===========================================================================

def align_financials_to_panel(
    financials: dict[str, list[FinancialMetrics]],
    panel: pd.DataFrame,
    field: str,
) -> pd.Series:
    """把 FinancialMetrics 的某个字段 forward-fill 对齐到行情 panel 的 (date, ticker).

    步骤（per ticker）：
        1. 把 list[FinancialMetrics] 转 (报告期, value) 的 Series，按报告期升序
        2. reindex 到 panel 该 ticker 的交易日，forward-fill
        3. 拼回 panel 全索引

    防未来函数：astock_source.get_financials 已保证 report_period <= end_date，
    所以「按 report_period 升序 + ffill」自然只用历史 —— 不会泄漏未来报告期。
    """
    # panel 的 ticker level 可能是 int（CSV roundtrip 把 '000858' 读成 858）或 str；
    # financials 的 key 是 str —— 两边做一个鲁棒匹配
    panel_tickers = panel.index.get_level_values("ticker").unique()
    ticker_map = _build_ticker_map(panel_tickers, list(financials.keys()))

    parts: list[pd.Series] = []
    for panel_tk in panel_tickers:
        fin_key = ticker_map.get(panel_tk)
        if fin_key is None:
            continue
        recs = financials.get(fin_key, [])
        if not recs:
            continue
        # 取该字段 + 报告期，剔除 None
        rows = [(r.report_period, getattr(r, field)) for r in recs if getattr(r, field) is not None]
        if not rows:
            continue
        # report_period 是 'YYYYMMDD' 字符串
        dates = pd.to_datetime([p for p, _ in rows], format="%Y%m%d", errors="coerce")
        vals = [v for _, v in rows]
        report_s = pd.Series(vals, index=dates, dtype=float).sort_index()
        # 去重：同一报告期出现多次取最后
        report_s = report_s[~report_s.index.duplicated(keep="last")]

        # 该 ticker 在 panel 上的所有交易日
        tk_idx = panel.xs(panel_tk, level="ticker").index
        # forward-fill 到交易日
        aligned = report_s.reindex(report_s.index.union(tk_idx)).sort_index().ffill().reindex(tk_idx)
        aligned.index = pd.MultiIndex.from_product([tk_idx, [panel_tk]], names=["date", "ticker"])
        parts.append(aligned)

    if not parts:
        return pd.Series(index=panel.index, dtype=float)

    out = pd.concat(parts).sort_index()
    return out.reindex(panel.index)


def _build_ticker_map(panel_tickers, fin_keys: list[str]) -> dict:
    """把 panel 的 ticker（可能是 int / str）映射到 financials 的 str key.

    P2 的 CSV 缓存把 '000858' 读成 int 858（丢了前导零），这里宽松匹配：
    int 858 ↔ str '000858' / '858' / '0000858'。
    """
    out: dict = {}
    for pt in panel_tickers:
        pt_str = str(pt)
        for fk in fin_keys:
            if fk.lstrip("0") == pt_str.lstrip("0"):
                out[pt] = fk
                break
    return out


# ===========================================================================
# 因子基类辅助：财务因子统一从 kwargs 取 financials 字典
# ===========================================================================

class _FundamentalBase(BaseFactor):
    """财务因子共用：从 kwargs['financials'] 取 dict，调 align_financials_to_panel.

    子类只需覆盖 name 和 _compute_from_aligned（拿到对齐后的 Series 再做计算）。
    """

    def compute(self, panel: pd.DataFrame, **kwargs) -> pd.Series:
        financials = kwargs.get("financials") or {}
        if not financials:
            return pd.Series(index=panel.index, dtype=float, name=self.name)
        return self._compute_inner(panel, financials)

    def _compute_inner(self, panel: pd.DataFrame, financials: dict) -> pd.Series:
        raise NotImplementedError


# ===========================================================================
# 1. 估值因子（PE / PB）
# ===========================================================================

class PE(_FundamentalBase):
    """PE（市盈率 TTM）—— 来自腾讯实时快照，挂在最新报告期上.

    Stage 1 警告：PE 历史值全是 None（astock_source 只附最新），所以这个因子
    在训练集上几乎全 NaN，模型学不到。要让 PE 真正可用需要补齐历史 PE 序列。

    负值处理：腾讯 PE 在亏损股上是负数（TTM 净利润 < 0）。负 PE 数值不可比，
    作为模型特征会污染 —— 直接置 NaN 让下游 mask 掉。
    """

    @property
    def name(self) -> str:
        return "pe"

    def _compute_inner(self, panel: pd.DataFrame, financials: dict) -> pd.Series:
        s = align_financials_to_panel(financials, panel, "pe")
        s = s.where(s > 0)  # 负 PE（亏损股）→ NaN
        s.name = self.name
        return s


class PB(_FundamentalBase):
    """PB（市净率）—— 来自腾讯实时快照，同 PE 警告.

    负 PB（深度资不抵债公司）极罕见但理论上存在，作为特征不可比 → NaN.
    """

    @property
    def name(self) -> str:
        return "pb"

    def _compute_inner(self, panel: pd.DataFrame, financials: dict) -> pd.Series:
        s = align_financials_to_panel(financials, panel, "pb")
        s = s.where(s > 0)
        s.name = self.name
        return s


# ===========================================================================
# 2. 盈利因子（ROE / 净利率）
# ===========================================================================

class ROE(_FundamentalBase):
    """ROE（净资产收益率，%）—— 直接取 FinancialMetrics.roe.

    每个报告期独立值，forward-fill 到交易日。akshare 把单位写成百分数（如 10.57 表示
    10.57%）—— 我们保持这个口径，模型只关心相对大小。
    """

    @property
    def name(self) -> str:
        return "roe"

    def _compute_inner(self, panel: pd.DataFrame, financials: dict) -> pd.Series:
        s = align_financials_to_panel(financials, panel, "roe")
        s.name = self.name
        return s


class NetMargin(_FundamentalBase):
    """净利率 = net_profit / revenue.

    需同时拿两个字段对齐到交易日后做除法。比 ROE 多一层「数据完整性依赖」——
    任一字段为 None 都会让结果 NaN。
    """

    @property
    def name(self) -> str:
        return "net_margin"

    def _compute_inner(self, panel: pd.DataFrame, financials: dict) -> pd.Series:
        net_profit = align_financials_to_panel(financials, panel, "net_profit")
        revenue = align_financials_to_panel(financials, panel, "revenue")
        s = self._safe_div(net_profit, revenue)
        s = self._replace_inf(s)
        s.name = self.name
        return s


class EPS(_FundamentalBase):
    """每股收益 —— 直接 forward-fill."""

    @property
    def name(self) -> str:
        return "eps"

    def _compute_inner(self, panel: pd.DataFrame, financials: dict) -> pd.Series:
        s = align_financials_to_panel(financials, panel, "eps")
        s.name = self.name
        return s


# ===========================================================================
# 3. 成长因子 —— 同比增长率
# ===========================================================================

class RevenueGrowthYoY(_FundamentalBase):
    """营收同比增速 = revenue[T] / revenue[T-4 期] - 1（4 个季度 ≈ 一年）.

    实现：对每只票，把 revenue 按报告期排序后做 pct_change(4)，再 forward-fill
    到交易日。这里直接在「报告期序列」上做 pct_change，再对齐 panel ——
    比先对齐后 pct_change 更准（避免日频 ffill 干扰季度差分）。
    """

    @property
    def name(self) -> str:
        return "revenue_growth_yoy"

    def _compute_inner(self, panel: pd.DataFrame, financials: dict) -> pd.Series:
        return _yoy_growth_factor(panel, financials, "revenue", self.name, self)


class NetProfitGrowthYoY(_FundamentalBase):
    """净利润同比增速 = net_profit[T] / net_profit[T-4] - 1.

    注意：净利润可能为负，pct_change 在负→正切换时会出现误导性的正负号 ——
    实务里常用「绝对值口径」或「同比改善方向」。Stage 1 用原始 pct_change，
    模型自己学这种非线性。
    """

    @property
    def name(self) -> str:
        return "net_profit_growth_yoy"

    def _compute_inner(self, panel: pd.DataFrame, financials: dict) -> pd.Series:
        return _yoy_growth_factor(panel, financials, "net_profit", self.name, self)


def _yoy_growth_factor(
    panel: pd.DataFrame,
    financials: dict,
    field: str,
    name: str,
    base: BaseFactor,
) -> pd.Series:
    """通用「同比增速」实现：对每只票按报告期 pct_change(4) 后对齐到 panel."""
    panel_tickers = panel.index.get_level_values("ticker").unique()
    ticker_map = _build_ticker_map(panel_tickers, list(financials.keys()))

    parts: list[pd.Series] = []
    for panel_tk in panel_tickers:
        fin_key = ticker_map.get(panel_tk)
        if fin_key is None:
            continue
        recs = financials.get(fin_key, [])
        rows = [(r.report_period, getattr(r, field)) for r in recs if getattr(r, field) is not None]
        if len(rows) < 5:  # 至少要有 5 期才能算 yoy
            continue

        dates = pd.to_datetime([p for p, _ in rows], format="%Y%m%d", errors="coerce")
        vals = [v for _, v in rows]
        report_s = pd.Series(vals, index=dates, dtype=float).sort_index()
        report_s = report_s[~report_s.index.duplicated(keep="last")]

        # 注意：A股 季报有时跳报告期（如只发年报），按位置 pct_change(4) 可能错配。
        # 实务中更准的做法是「同月份对比」，Stage 1 简化为位置 pct_change（够用）。
        yoy = report_s.pct_change(4)

        tk_idx = panel.xs(panel_tk, level="ticker").index
        aligned = yoy.reindex(yoy.index.union(tk_idx)).sort_index().ffill().reindex(tk_idx)
        aligned.index = pd.MultiIndex.from_product([tk_idx, [panel_tk]], names=["date", "ticker"])
        parts.append(aligned)

    if not parts:
        return pd.Series(index=panel.index, dtype=float, name=name)

    out = pd.concat(parts).sort_index().reindex(panel.index)
    out = base._replace_inf(out)
    out.name = name
    return out


__all__ = [
    "PE",
    "PB",
    "ROE",
    "NetMargin",
    "EPS",
    "RevenueGrowthYoY",
    "NetProfitGrowthYoY",
    "align_financials_to_panel",
]
