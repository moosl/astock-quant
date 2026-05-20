"""A股 新闻情绪 LLM prompt 模板.

借鉴 .p0-repos/TradingAgents-astock/agents/analysts/social_media_analyst.py +
news_analyst.py 的「A股 情绪/政策分析框架」：
- A股 散户占比 > 60%，情绪波动剧烈、对短期价格影响远大于成熟市场
- 政策市：央行 / 证监会 / 发改委政策直接定方向
- 板块轮动：单条利好可能带动整个产业链

我们的场景比这两个分析师简单得多 —— 不是写长篇报告，而是**对单条新闻打一个情绪分**。
所以 prompt 也要瘦身：
- 不要 BUY/HOLD/SELL 最终建议（那是模型 + 回测的事，不是 LLM 因子）
- 不要 Markdown 报告（要纯 JSON）
- 不要 tool_calls（我们直接喂新闻文本）

输出契约：见 schemas.py 的 NewsSentimentOutput。
"""

from __future__ import annotations

# ===========================================================================
# 单条新闻情绪打分 prompt
# ===========================================================================

NEWS_SENTIMENT_SYSTEM = """你是一位 A股 市场新闻情绪分析师。任务：给单条新闻打一个对目标个股的情绪分。

【A股 情绪分析要点】
- 散户占比超 60%，情绪对短期价格影响远大于成熟市场，情绪因子有效。
- 政策市：央行 / 证监会 / 发改委政策对市场影响显著，优先识别政策类新闻的方向。
- 板块联动：行业利好可能带动产业链，行业利空同理。
- 区分官方消息（权威）vs 市场传闻（噪音）。
- 业绩预告 / 业绩快报 / 重大合同 / 股东大会 / 监管处罚等是常见事件驱动信号。

【情绪五级】
- very_negative (-1.0)：明显利空，可能导致股价显著下跌（业绩暴雷、监管处罚、重大风险事件等）
- negative      (-0.5)：温和利空，对股价有负面影响（业绩低于预期、行业政策收紧等）
- neutral       ( 0.0)：中性 / 信息不足 / 既非利好也非利空 / 无明显影响
- positive      (+0.5)：温和利好，对股价有正面影响（业绩超预期、新合同、行业利好等）
- very_positive (+1.0)：明显利好，可能导致股价显著上涨（重大政策利好、并购重组、技术突破等）

【输出要求】
- 严格按 JSON 格式输出，不要任何其它文字、不要 Markdown 代码块。
- JSON 必须包含三个字段：
  - sentiment: 数字，范围 [-1.0, +1.0]，按五级取 -1/-0.5/0/+0.5/+1。
  - confidence: 数字，[0, 1]，对自己判断的信心。信息不足、关联弱时调低。
  - reason: 字符串，一句话说明理由（不超过 50 字）。

【信息不足时的处理】
- 标题 / 内容看不出和目标个股的明确关联 → sentiment=0.0, confidence 调低 (< 0.3)
- 完全没法判断 → sentiment=0.0, confidence=0.0, reason="信息不足"
"""


def build_news_sentiment_user_prompt(
    ticker: str,
    title: str,
    content: str,
    source: str = "",
    date: str = "",
) -> str:
    """构造单条新闻情绪打分的 user message.

    精简 —— 把目标个股、新闻关键字段拼成一段，让 LLM 看得清，不堆废话。

    参数：
        ticker:  目标个股 6 位代码（如 "600519"）
        title:   新闻标题
        content: 新闻正文 / 摘要（可空）
        source:  来源（如 "东方财富"）
        date:    发布日期（YYYY-MM-DD，可空）

    返回：
        user message 字符串，直接拼到 messages 里。
    """
    # 把过长的正文截断 —— 防止单条新闻吃掉 8K+ token；前 2000 字符足够 LLM 判断方向
    content_truncated = (content or "")[:2000]

    parts = [f"目标个股：{ticker}"]
    if date:
        parts.append(f"发布日期：{date}")
    if source:
        parts.append(f"来源：{source}")
    parts.append(f"标题：{title}")
    if content_truncated:
        parts.append(f"内容：{content_truncated}")

    parts.append("\n请按系统消息要求，输出 JSON。")
    return "\n".join(parts)


__all__ = ["NEWS_SENTIMENT_SYSTEM", "build_news_sentiment_user_prompt"]
