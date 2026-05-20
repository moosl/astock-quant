"""数据源抽象 —— 所有数据源都实现的接口（结构化类型，无需继承）.

思路来自 ai-hedge-fund v2 data/protocol.py：用 typing.Protocol 定义「数据源长什么样」，
任何实现了这几个方法的类都自动满足 DataSource —— 不需要显式继承。
这是「换数据源」和「Stage 2 接文本源」的扩展位。

约定（与 v2 protocol 一致）：
- 方法失败返回空 list，不抛异常 —— 让上层（dataset / cache）决定怎么处理缺数据。
- 返回的都是 contracts.py 的 Pydantic 对象 list，不返回裸 DataFrame / dict。
- ticker 一律纯 6 位代码字符串（如 "600519"），日期一律 "YYYY-MM-DD" 字符串入参。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astock_quant.contracts import (
    FinancialMetrics,
    MoneyFlowRecord,
    NewsItem,
    PriceBar,
)


@runtime_checkable
class DataSource(Protocol):
    """数据源 Protocol —— A股 数据适配器（及未来其它源）实现此接口.

    4 个方法对应 4 类原始数据：行情 / 财务 / 资金流 / 文本。
    Stage 1 的因子层只用到 get_prices（量价因子）和 get_financials（财务因子）+
    get_moneyflow（资金流因子）；get_news 的形状 Stage 1 就固定，Stage 2 LLM 因子用。
    """

    def get_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[PriceBar]:
        """拉取 [start_date, end_date] 区间的日线 OHLCV.

        返回按日期升序的 PriceBar list。失败 / 无数据返回 []。
        """
        ...

    def get_financials(
        self, ticker: str, end_date: str
    ) -> list[FinancialMetrics]:
        """拉取截至 end_date 的财务指标（防未来函数：不返回 end_date 之后发布的报告期）.

        返回 FinancialMetrics list（可能多期，按报告期升序）。失败 / 无数据返回 []。
        """
        ...

    def get_moneyflow(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[MoneyFlowRecord]:
        """拉取 [start_date, end_date] 区间的资金流记录.

        注意部分端点只有短历史（北向断供、个股资金流仅近 20 日），
        返回的记录可能远短于请求区间。失败 / 无数据返回 []。
        """
        ...

    def get_news(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[NewsItem]:
        """拉取 [start_date, end_date] 区间的个股新闻文本 —— Stage 2 LLM 因子入口.

        Stage 1 就实现（端点现成），但下游 Stage 2 才真正消费。
        返回 NewsItem list。失败 / 无数据返回 []。
        """
        ...
