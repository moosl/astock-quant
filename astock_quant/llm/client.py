"""LLM 客户端抽象 + 多 provider 实现 + 工厂函数.

设计目标（方案 A 瘦身版）：
- 用 typing.Protocol 抽象一个**最小**客户端接口（chat / chat_json）—— 不继承 ABC，
  任何实现这俩方法的类都自动满足 Protocol。
- 默认 provider 是本机 Codex CLI（复用登录态，不需要模型 API key）。
- 切 OpenAI / DeepSeek / Kimi 等只需新增一个 implementer class + factory 里加一行，
  调用方一律走 `make_llm_client(provider)`。
- 结构化输出 helper：让 LLM 吐 JSON，Pydantic 解析；失败时 fail loud（让因子层决定怎么处理）。

API key 安全：
- 一律从 env var 读，**绝不**硬编码、**绝不**写进 docstring 示例。
- Codex CLI 复用本机登录态；兼容 provider 缺 key 时由上游决定降级 / 跳过。

参考代码：.p0-repos/TradingAgents-astock/llm_clients/ 的多供应商抽象（重抽 langchain
ChatModel 兼容层；我们这层瘦身得只剩两个方法）。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ===========================================================================
# 异常
# ===========================================================================

class LLMClientError(RuntimeError):
    """LLM 客户端错误 —— API key 缺失 / 调用失败 / 解析失败统一抛这个.

    上游（因子层）抓到这个可以决定：
    - 缺 key → 因子全 NaN（默认行为：因子层故意不在初始化时强求 key）
    - 调用失败 → 跳过当条新闻、记 warning
    - 解析失败 → 跳过、记 warning + 原始文本
    """


# ===========================================================================
# 响应 + Protocol
# ===========================================================================

@dataclass
class LLMResponse:
    """LLM 一次调用的结构化结果.

    `content` 是 LLM 吐的纯文本（如果是 chat_json，content 是原始 JSON 字符串便于 debug）。
    `usage` 是 token 用量（不强求，部分 provider 可能给不到）。
    """
    content: str
    usage: dict[str, int] | None = None
    model: str = ""


@runtime_checkable
class LLMClient(Protocol):
    """LLM 客户端 Protocol —— 任何 provider 的实现都长这样.

    最小契约：两个方法。
    - chat：自由文本输入输出（messages → 字符串）
    - chat_json：结构化输出（messages + schema → Pydantic 实例）

    各 provider 内部细节（auth / endpoint / 重试 / 流式）不暴露给调用方。
    """

    provider: str
    model: str

    def chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """普通对话调用. messages 是 [{"role": "user"/"assistant", "content": "..."}, ...]."""
        ...

    def chat_json(
        self,
        messages: list[dict],
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> T:
        """结构化输出调用：让 LLM 吐 JSON、用 schema 解析成 Pydantic 实例.

        实现策略：在 system / user prompt 里加「严格 JSON 格式」要求，
        然后调 chat 拿到字符串，正则提取 JSON 块，pydantic validate。
        失败抛 LLMClientError，附原始文本便于 debug。
        """
        ...


# ===========================================================================
# 默认实现：Anthropic Claude
# ===========================================================================

class AnthropicClient:
    """Anthropic Claude 客户端 —— 默认 provider.

    直接用 `anthropic` 官方 SDK，不引 langchain。
    env var：`ANTHROPIC_API_KEY`（必需）、`ANTHROPIC_BASE_URL`（可选，中转站用）。
    模型默认 claude-haiku-4-5（因子打分场景对成本敏感、不需要顶配模型）；可通过
    `LLM_MODEL` env var 覆盖。

    用法：
        client = AnthropicClient()  # 走 env var
        resp = client.chat([{"role": "user", "content": "你好"}])
        out = client.chat_json(messages, schema=NewsSentimentOutput)
    """

    provider: str = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # api_key：参数优先，否则 env var；都没有则 fail loud
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMClientError(
                "ANTHROPIC_API_KEY 未设置 —— Anthropic client 需要 API key。"
                " 请 export ANTHROPIC_API_KEY=<your-key>，或切换 provider。"
            )

        # 模型：参数 > env var LLM_MODEL > 默认 haiku 4.5（因子打分性价比最高）
        self.model = model or os.environ.get("LLM_MODEL") or "claude-haiku-4-5"
        self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")

        # 延迟 import anthropic —— 测试场景（mock client）不会引这个依赖
        try:
            import anthropic
        except ImportError as e:
            raise LLMClientError(
                "anthropic 包未安装。运行：uv add anthropic"
            ) from e

        client_kwargs = {"api_key": key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._sdk = anthropic.Anthropic(**client_kwargs)

    # ----------------------------------------------------------------------
    # chat
    # ----------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Anthropic messages API 一次调用 —— 返回纯文本."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system:
                kwargs["system"] = system
            resp = self._sdk.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 —— SDK 异常包成我们的统一类型
            raise LLMClientError(f"Anthropic 调用失败: {e}") from e

        # resp.content 是 list[ContentBlock]，TextBlock.text 拼起来
        text = "".join(
            getattr(block, "text", "")
            for block in (resp.content or [])
            if getattr(block, "type", "") == "text"
        )
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", 0),
                "output_tokens": getattr(resp.usage, "output_tokens", 0),
            }
        return LLMResponse(content=text, usage=usage, model=self.model)

    # ----------------------------------------------------------------------
    # chat_json
    # ----------------------------------------------------------------------

    def chat_json(
        self,
        messages: list[dict],
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> T:
        """结构化输出：拿纯文本 → 提取 JSON → Pydantic 解析."""
        resp = self.chat(
            messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return parse_json_to_schema(resp.content, schema)


# ===========================================================================
# JSON 解析 helper（共享 —— 不同 provider 的 chat_json 都用这个）
# ===========================================================================

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_to_schema(text: str, schema: type[T]) -> T:
    """从 LLM 返回文本里抽 JSON、用 schema 解析.

    LLM 有时会包 Markdown 代码块或在前后加废话，所以先用正则抠 `{...}` 最长匹配，
    然后 json.loads + schema.model_validate。失败一律抛 LLMClientError（附原始文本）。

    这个函数公开 —— 测试里也复用。
    """
    if not text or not text.strip():
        raise LLMClientError(f"LLM 返回空文本，无法解析 {schema.__name__}")

    # 优先尝试直接 parse（最快路径：LLM 严格按要求只吐 JSON）
    candidates: list[str] = [text.strip()]

    # 后备：正则抠最长 {...}（应对 LLM 加 Markdown / 前后废话）
    match = _JSON_BLOCK_RE.search(text)
    if match:
        candidates.append(match.group(0))

    last_err: Exception | None = None
    for raw in candidates:
        try:
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            continue

    raise LLMClientError(
        f"LLM 返回无法解析为 {schema.__name__}: {last_err}\n原始文本: {text[:500]}"
    )


# ===========================================================================
# 工厂
# ===========================================================================

# 支持的 provider 注册表；Codex CLI 在文件末尾注册，避免循环 import。
_PROVIDERS: dict[str, type] = {
    "anthropic": AnthropicClient,
}


def make_llm_client(provider: str | None = None, **kwargs) -> LLMClient:
    """LLM 客户端工厂.

    参数：
        provider: provider 名（"codex" / "anthropic" / "deepseek"）。
                 None 时读 env var `LLM_PROVIDER`，仍为 None 则默认 "codex"。
        kwargs:  透传给具体 client 构造函数（model / api_key / base_url 等）。

    抛 LLMClientError：provider 不支持、API key 未设置、SDK 未安装。

    用法：
        # 默认 Codex CLI，复用本机登录态
        client = make_llm_client()

        # 显式指定 provider 和模型
        client = make_llm_client("anthropic", model="claude-opus-4-7")
    """
    p = (provider or os.environ.get("LLM_PROVIDER") or "codex").lower()
    impl_cls = _PROVIDERS.get(p)
    if impl_cls is None:
        raise LLMClientError(
            f"不支持的 LLM provider: {p}。"
            f" 已支持: {sorted(_PROVIDERS.keys())}。"
            f" 新增请改 astock_quant/llm/client.py 的 _PROVIDERS 注册表。"
        )
    return impl_cls(**kwargs)


__all__ = [
    "AnthropicClient",
    "LLMClient",
    "LLMClientError",
    "LLMResponse",
    "make_llm_client",
    "parse_json_to_schema",
]


# ===========================================================================
# 后注册的 provider —— 必须放在文件末尾以避免循环 import
# ===========================================================================
#
# `deepseek.py` 顶层 `from astock_quant.llm.client import LLMClientError,
# LLMResponse, parse_json_to_schema`；如果我们在 client.py 顶部 import deepseek 会
# 产生循环（client → deepseek → client 还没加载完）。
#
# 把 DeepSeekClient 的 import + 注册放在文件**末尾**：到这行执行时 client.py 的所有
# 顶层对象（LLMClientError / LLMResponse / parse_json_to_schema / _PROVIDERS）都
# 已就位，deepseek.py 可以安全 import 它们。
#
# 这是 P7 启用的 provider；anthropic 的注册保持在上方原位置不动（向后兼容）。
from astock_quant.llm.deepseek import DeepSeekClient  # noqa: E402
from astock_quant.llm.codex_cli import CodexCLIClient  # noqa: E402

_PROVIDERS["deepseek"] = DeepSeekClient
_PROVIDERS["codex"] = CodexCLIClient
