"""信号层 —— 模型预测 → 人可读的买卖 / 持仓信号.

generator 把模型 Prediction 转成 SignalReport。4 类预测目标共用，按 target_type 分派。
"""
