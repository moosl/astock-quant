"""回测层 —— 逐日回测引擎 + A股约束 + 绩效指标.

engine 逐日推进、portfolio 管持仓现金、constraints 加 A股专属规则、metrics 算绩效。
核心逻辑重度参考 ai-hedge-fund v1 src/backtesting/（研读后用自己的话重写，不直接 import）。
产出 contracts.py 的 BacktestResult。
"""
