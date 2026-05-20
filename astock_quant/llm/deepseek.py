"""DeepSeek LLM 客户端 —— P7 启用，OpenAI 兼容协议.

为什么加 DeepSeek：
- 用户决定 P7 改用 DeepSeek（比 Anthropic 便宜 ~10x，token 大量打分场景敏感）
- DeepSeek API 完全 OpenAI 兼容（chat/completions、response_format JSON 模式都和 OpenAI 一样）
- 不引 `openai` SDK —— 直接 httpx 调 https://api.deepseek.com，更轻
  (httpx 是 anthropic SDK 的传递依赖，已经在装好的环境里，不加新顶层依赖)

实现思路（保持 P6 的方案 A 瘦身版纪律）：
- 直接 httpx 调 chat/completions 端点
- 默认 model = `deepseek-v4-pro`（用户指定）
- chat_json 用 `response_format={"type": "json_object"}` 强制 JSON 输出
  + 沿用 `parse_json_to_schema` helper 做兜底（应对 LLM 偶尔吐 markdown 包装）

env var：
- `DEEPSEEK_API_KEY`（必需）
- `LLM_MODEL`（可选，默认 deepseek-v4-pro；deepseek-v4-flash 也可用，便宜更多）
- `DEEPSEEK_BASE_URL`（可选，自建代理 / 中转站）

满足 LLMClient Protocol：实现 `chat` + `chat_json` 两个方法即可，无需继承任何基类
（Protocol 是结构化类型，对鸭子类型友好）。
"""

from __future__ import annotations

import json
import logging
import os
from typing import TypeVar

import httpx
from pydantic import BaseModel

from astock_quant.llm.client import (
    LLMClientError,
    LLMResponse,
    parse_json_to_schema,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# DeepSeek 官方 base URL（2026-05 时验证）—— 走 OpenAI 兼容路径
_DEFAULT_BASE_URL = "https://api.deepseek.com"

# 默认模型 = 用户指定的 deepseek-v4-pro
_DEFAULT_MODEL = "deepseek-v4-pro"

# 默认 HTTP 超时（秒）—— LLM 打分场景偏长，60s 足够，不卡死
_DEFAULT_TIMEOUT = 60.0


class DeepSeekClient:
    """DeepSeek 客户端 —— P7 默认 provider.

    构造：从 env var 读 key + 可选 base_url override。
    用法（同 AnthropicClient 一样满足 LLMClient Protocol）：
        client = DeepSeekClient()  # 走 env var DEEPSEEK_API_KEY
        resp = client.chat([{"role": "user", "content": "你好"}])
        out = client.chat_json(messages, schema=NewsSentimentOutput)
    """

    provider: str = "deepseek"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        # API key：参数优先 → env var → fail loud
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise LLMClientError(
                "DEEPSEEK_API_KEY 未设置 —— DeepSeek client 需要 API key。"
                " 请 export DEEPSEEK_API_KEY=<your-key>，或切换 provider。"
            )
        self._api_key = key

        # 模型：参数 > env var LLM_MODEL > 默认 deepseek-v4-pro
        self.model = model or os.environ.get("LLM_MODEL") or _DEFAULT_MODEL

        # base URL：参数 > env var > 官方默认
        self.base_url = (
            base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")

        self.timeout = timeout or _DEFAULT_TIMEOUT

    # ----------------------------------------------------------------------
    # 内部：构造 + 发送 chat/completions 请求
    # ----------------------------------------------------------------------

    def _request_chat_completion(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict | None = None,
    ) -> dict:
        """发送 chat/completions POST，返回原始 JSON dict.

        DeepSeek 端点和 OpenAI 完全一致：
          POST {base_url}/chat/completions
          Authorization: Bearer {key}
          body: {model, messages, temperature, max_tokens, response_format?}

        失败统一抛 LLMClientError（HTTP 非 2xx / 网络错误 / 解析失败）。
        如果 DeepSeek 返回「model_not_found」之类的明确错误，错误信息会回到上层，
        让 verifier / 用户知道实际可用的模型名。
        """
        url = f"{self.base_url}/v1/chat/completions"

        # OpenAI 协议把 system 作为 messages 里第一条 {"role": "system"}
        # 不是顶层 system 字段（那是 Anthropic 风格）
        full_messages: list[dict] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        body: dict = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout) as h:
                resp = h.post(url, headers=headers, json=body)
        except httpx.HTTPError as e:
            raise LLMClientError(f"DeepSeek HTTP 调用失败: {e}") from e

        if resp.status_code != 200:
            # 把 DeepSeek 返回的 error.message 透传给上层，便于排错（如 model_not_found）
            err_detail = _safe_extract_error(resp.text)
            raise LLMClientError(
                f"DeepSeek HTTP {resp.status_code}: {err_detail}"
            )

        try:
            return resp.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise LLMClientError(
                f"DeepSeek 返回非 JSON 响应: {resp.text[:300]}"
            ) from e

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
        """OpenAI-兼容 chat completions（无 JSON 强制）—— 返回纯文本."""
        data = self._request_chat_completion(
            messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _normalize_openai_response(data, self.model)

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
        """结构化输出：强制 response_format=json_object + Pydantic 解析.

        DeepSeek JSON 模式要求：
          - response_format = {"type": "json_object"}
          - prompt 里需出现 "json" 字眼 + 提供格式示例（我们的 prompts.py 已满足）
          - max_tokens 不要太小（防截断）

        失败抛 LLMClientError 由上游（factor 层）决定怎么处理（跳过这条新闻 / 全 NaN）。
        """
        data = self._request_chat_completion(
            messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        resp = _normalize_openai_response(data, self.model)
        if not resp.content or not resp.content.strip():
            raise LLMClientError(
                "DeepSeek 返回空 content —— 提示词可能没含 'json' 字眼或被截断"
            )
        return parse_json_to_schema(resp.content, schema)


# ===========================================================================
# Helpers —— OpenAI 协议响应规整 / 错误抽取
# ===========================================================================

def _normalize_openai_response(data: dict, model: str) -> LLMResponse:
    """把 OpenAI chat/completions 响应规整成 LLMResponse.

    标准 schema：
      data["choices"][0]["message"]["content"]: str
      data["usage"]: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    """
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMClientError(
            f"DeepSeek 响应缺 choices[0].message.content: {str(data)[:200]}"
        ) from e

    usage = None
    raw_usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    if raw_usage:
        usage = {
            "input_tokens": int(raw_usage.get("prompt_tokens", 0)),
            "output_tokens": int(raw_usage.get("completion_tokens", 0)),
        }
    return LLMResponse(content=content, usage=usage, model=model)


def _safe_extract_error(text: str) -> str:
    """从 DeepSeek 错误响应中抽出可读的 error.message，失败则返回原文截断."""
    if not text:
        return "<empty body>"
    try:
        blob = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return text[:300]
    if isinstance(blob, dict):
        err = blob.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("type") or ""
            code = err.get("code") or ""
            if msg:
                return f"{msg}" + (f" (code={code})" if code else "")
        # 兜底：直接给整个 dict 的字符串
        return json.dumps(blob, ensure_ascii=False)[:300]
    return text[:300]


__all__ = ["DeepSeekClient"]
