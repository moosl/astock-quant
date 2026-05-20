"""LLM 结构化输出 schemas —— 让 LLM 吐 JSON，Pydantic 解析.

参考 .p0-repos/TradingAgents-astock/agents/utils/structured.py 的思路（用自己的话
重写到极简）：让 LLM 返回结构化 JSON 而不是自由文本，下游不用写正则解析、
不会被废话淹没。

Stage 2 P6 只用 NewsSentimentOutput；将来加事件/政策因子时新增 schema 即可。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NewsSentimentOutput(BaseModel):
    """单条新闻的情绪 LLM 输出 —— P6 LLM 因子的「LLM 一次调用产物」.

    五级情绪映射到 [-1, +1] 连续分（参考 social_media_analyst 的「极度悲观 / 悲观 /
    中性 / 乐观 / 极度乐观」框架）：
      - very_negative → -1.0   利空，强烈负面影响
      - negative      → -0.5   利空，温和负面影响
      - neutral       →  0.0   中性 / 无明显影响 / 信息不足
      - positive      → +0.5   利好，温和正面影响
      - very_positive → +1.0   利好，强烈正面影响

    `confidence` 是模型对自己判断的信心（0-1）。聚合层可用它做加权（不强制）。
    `reason` 一句话说明为什么这样打分 —— **debug 用**，不进缓存的核心字段，
    但落盘留底，方便回头看「为什么这条新闻被打了 -1」。
    """

    sentiment: float = Field(
        description="情绪分数，范围 [-1.0, +1.0]，按五级映射（见模块 docstring）"
    )
    confidence: float = Field(
        default=0.5,
        description="判断信心 [0, 1]，0=完全不确定，1=非常确定",
    )
    reason: str = Field(
        default="",
        description="一句话理由（debug 用，可留空）",
    )


__all__ = ["NewsSentimentOutput"]
