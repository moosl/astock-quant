"""因子层 —— 原始数据 → 因子值.

BaseFactor (ABC) ← price_volume / fundamental / moneyflow / llm_factor 各自实现
registry 批量计算编排 → 产出 FactorFrame（模型层的输入特征 X）。

关键：LLM 因子（llm_factor）和量价因子在架构里平级 —— 同一个父类 BaseFactor、
同样产出 FactorValue、同经 registry 汇成 FactorFrame。对下游完全无差别，
这是 Stage 2 LLM 因子能平滑接入的根本原因。
"""
