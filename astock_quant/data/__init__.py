"""数据层 —— 拉数 + 适配 + 构建数据集.

DataSource (protocol) ← astock_source 实现 → cache 缓存/截断 → dataset 构建 panel
对外产出 contracts.py 的 PriceBar / FinancialMetrics / MoneyFlowRecord / NewsItem。
"""
