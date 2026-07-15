"""LLM 多供应商客户端封装 —— P6 LLM 情绪因子的基础设施.

设计要点（方案 A 瘦身版）：
- 研读 .p0-repos/TradingAgents-astock/llm_clients/ 的多供应商抽象（默认抽 LangChain
  ChatModel），**不 import 任何框架** —— 自己用 Protocol 抽象一个最小客户端接口。
- 默认 provider 是本机 Codex CLI（复用登录态，不需要模型 API key）。
- 切 OpenAI / DeepSeek / Kimi 等只需新增一个 implementer class，不动调用方 ——
  调用方一律走 `make_llm_client(provider)` 工厂。
- 结构化输出：让 LLM 吐 JSON，用 Pydantic 解析。失败时降级到自由文本 + 正则提取。

API key 安全：
- 兼容 provider 的 key 一律从环境变量读，**绝不**硬编码、**绝不**写进缓存文件。
- Codex CLI 复用本机登录态；兼容 provider 缺 key 时由上游决定降级 / 跳过。

入口：
    from astock_quant.llm import make_llm_client, LLMClient
    from astock_quant.llm.schemas import NewsSentimentOutput

    client = make_llm_client()  # 默认 Codex CLI
    result: NewsSentimentOutput = client.chat_json(
        messages=[{"role": "user", "content": "..."}],
        schema=NewsSentimentOutput,
    )
"""

from astock_quant.llm.client import (
    AnthropicClient,
    LLMClient,
    LLMClientError,
    LLMResponse,
    make_llm_client,
)
from astock_quant.llm.deepseek import DeepSeekClient
from astock_quant.llm.codex_cli import CodexCLIClient
from astock_quant.llm.schemas import NewsSentimentOutput

__all__ = [
    "AnthropicClient",
    "CodexCLIClient",
    "DeepSeekClient",
    "LLMClient",
    "LLMClientError",
    "LLMResponse",
    "make_llm_client",
    "NewsSentimentOutput",
]
