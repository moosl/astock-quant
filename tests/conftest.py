"""pytest 全局配置 —— 测试隔离防 .env 污染.

为什么需要：
- `astock_quant/__init__.py` 在 package import 时自动 load `.env`（Stage 2 P6/P7 跑 LLM 因子的便利设计）
- 但开发机 `.env` 通常会有 `ENABLE_LLM_FACTOR=1` / `DEEPSEEK_API_KEY=sk-...` / `LLM_PROVIDER=deepseek`
- 测试默认必须跑在「无 LLM」基线（避免误打真实 API 烧钱、保持 default_factors() 数量稳定、保持 fail-loud 行为可测）
- 需要 LLM 行为的测试自己用 `monkeypatch.setenv` 显式打开（`test_llm_factor_with_mock_client.py` 已是这模式）

实现：autouse fixture 在每个 test function 开始前清空这些 env var；
monkeypatch 在 function teardown 自动回滚，不影响其他 session。
"""

from __future__ import annotations

import pytest

# 测试基线必须清空的 LLM 相关 env var
# —— 任何被 astock_quant/__init__.py 的 load_dotenv 注入的、可能影响测试默认行为的变量
_LLM_ENV_VARS_TO_ISOLATE = (
    "ENABLE_LLM_FACTOR",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
)


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    """每个测试开始前清空 LLM 相关 env var，让测试默认跑在「无 LLM」基线.

    需要 LLM 行为的测试在函数体内继续用 monkeypatch.setenv 显式开启，
    会覆盖本 fixture 的 delenv（同一个 monkeypatch 实例，后写胜出）。
    """
    for var in _LLM_ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
