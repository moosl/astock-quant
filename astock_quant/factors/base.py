"""因子基类 —— 所有因子的统一接口.

思路来自 ai-hedge-fund v2 signals/base.py 的 BaseSignal ABC（研读后用自己的话重写）。
所有因子（量价 / 财务 / 资金流 / Stage 2 LLM）继承 BaseFactor，
产出 per-(date, ticker) 的因子值 —— 由 registry 汇集成 FactorFrame。

设计要点：
- 每个因子都是「自包含 + 无状态」—— `compute(panel)` 只看输入，不存中间状态。
- 因子产出 pd.Series（MultiIndex=(date, ticker)，单列），registry 把多个因子拼成
  FactorFrame.data 的多列。这是「单因子写起来简单」「批量计算高效」的折中。
- 防 look-ahead：因子内部所有窗口必须只看历史（如 `rolling(20)` 看过去 20 日）。
  数据进入因子前已经过 cache.truncate_by_date 截断（panel 不含未来），因子内部
  再用 `.shift(0)` 自然只看历史，无需特殊处理。但「未来 N 日收益」这类必须丢给
  labels/ 层，**因子层绝不允许 forward-looking 窗口**。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseFactor(ABC):
    """因子抽象基类。子类必须实现 `name` 和 `compute`。

    输入约定（panel）：
        DataFrame，MultiIndex=(date, ticker)，列 = open/high/low/close/volume/amount
        （由 data/dataset.build_price_panel 产出）。
        财务因子接收 dict[ticker, list[FinancialMetrics]] + 行情 panel 用于日期对齐；
        资金流因子接收资金流 panel（结构同行情）。

    输出约定：
        pd.Series，MultiIndex=(date, ticker)，name=self.name，dtype=float。
        缺失值用 NaN（不抛异常）—— 由 registry / 模型层决定怎么处理。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """因子标识（如 'momentum_20d', 'roe', 'main_inflow_5d'）。"""
        ...

    @abstractmethod
    def compute(self, panel: pd.DataFrame, **kwargs) -> pd.Series:
        """对 panel 计算因子值.

        参数：
            panel:   行情 panel（量价 / 资金流因子）或具体子类自定义入参。
            kwargs:  各因子可能用到的额外上下文（如 financials 字典）。

        返回：
            pd.Series，MultiIndex=(date, ticker)，name=self.name。NaN 表示数据不足 /
            缺失，不抛异常。
        """
        ...

    # ------------------------------------------------------------------
    # 共享 helper —— 子类按需用，统一 NaN / 极值处理口径
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        """安全除法：分母为 0 / NaN 时返回 NaN，避免 inf 污染下游。"""
        return num.div(den.replace(0, np.nan))

    @staticmethod
    def _winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """截尾去极值：把 < lower 分位和 > upper 分位的值压到对应分位 边界.

        ⚠️ 警告 —— 这是数据依赖型变换，**不能在因子层使用**.
        理由：分位 `lo / hi` 来自整列 `s.quantile(...)`，截断 panel vs 全量 panel
        的分位边界不同，会在切点产生非零差异（数据依赖型 look-ahead）。P3a 审核
        实测：在 net_margin / revenue_growth_yoy / net_profit_growth_yoy 上出现
        max abs diff 高达 0.47 的偏差。

        因子层应该只产出「原始数学定义值 + replace_inf + 合法 mask」，把截尾下沉到
        labels / preprocessing 层。需要在因子层做截尾时，必须改用：
          - **横截面分位**（按 date 分组、在当日 cross-section 内 quantile），或
          - **expanding 分位**（≤ T 的历史数据算分位，不看未来），
        而不是「整列后处理分位」。

        函数本体保留 —— 供 labels / preprocessing 层将来在「训练集 fit、验证集
        apply」的纪律下使用。
        """
        if s.empty or s.isna().all():
            return s
        lo, hi = s.quantile([lower, upper])
        if pd.isna(lo) or pd.isna(hi):
            return s
        return s.clip(lower=lo, upper=hi)

    @staticmethod
    def _replace_inf(s: pd.Series) -> pd.Series:
        """inf / -inf 一律替换为 NaN —— 防止下游模型遇到无穷大爆炸。"""
        return s.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _group_by_ticker(panel: pd.DataFrame, col: str) -> "pd.core.groupby.SeriesGroupBy":
        """对 panel 的某列按 ticker 分组，供 rolling / pct_change 等时序操作.

        panel 是 MultiIndex=(date, ticker)，按 ticker 分组后每组就是「单只票的时间序列」，
        可以安全做 rolling / shift —— 避免跨 ticker 串数据。
        """
        return panel[col].groupby(level="ticker", group_keys=False)
