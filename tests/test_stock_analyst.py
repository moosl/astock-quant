"""stock_analyst.py 测试 —— 全部 mock LLM client 和 SKILL adapter.

不打真网络, 不调真 LLM. 验证:
- analyze_stock 正常路径返回 markdown + disclaimer + 元数据
- LLM 失败时返回 fallback dict (markdown=None)
- pre_fetched 时不再调 SKILL adapter
- factor_context 字段进入 prompt
- market_overview 正常路径 / 失败兜底
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from astock_quant.llm.client import LLMClientError, LLMResponse
from astock_quant.llm.stock_analyst import (
    DISCLAIMER,
    analyze_stock,
    market_overview,
)


# ===========================================================================
# 共用 fixture
# ===========================================================================


@pytest.fixture
def mock_endpoints():
    """模拟 _fetch_endpoints_for_value 的返回."""
    return {
        "quote": {
            "600519": {
                "name": "贵州茅台", "price": 1500.0, "pe_ttm": 25.5, "pb": 9.2,
                "mcap_yi": 19000, "turnover_pct": 0.8,
            },
        },
        "info": {
            "code": "600519", "name": "贵州茅台", "industry": "白酒Ⅲ",
            "total_shares": 1.26e9, "mcap": 1.9e12, "list_date": "20010827",
        },
        "dividends": [
            {"date": "2024-06-30", "bonus_rmb": 30.0, "bonus_ratio": 0, "transfer_ratio": 0},
        ],
        "holders": [
            {"date": "2026-03-31", "holder_num": 100000, "change_ratio": -2.0},
        ],
        "fund_flow": [
            {"date": "2026-05-17", "main_net": 1e8, "small_net": 0, "mid_net": 0,
             "large_net": 0, "super_net": 0},
        ] * 20,
        "news": [
            {"title": "茅台一季报利润+15%", "content": "...", "time": "2026-04-30",
             "source": "财经网", "url": ""},
        ],
    }


@pytest.fixture
def mock_llm_client():
    """模拟 DeepSeek client.chat() 返回."""
    client = MagicMock()
    client.chat.return_value = LLMResponse(
        content="这是一段 AI 生成的价值分析。\n\n估值水平偏低, ROE 行业第一。",
        usage={"input_tokens": 500, "output_tokens": 200},
        model="deepseek-v4-pro",
    )
    return client


# ===========================================================================
# analyze_stock 正常路径
# ===========================================================================


class TestAnalyzeStock:
    def test_returns_required_keys(self, mock_endpoints, mock_llm_client):
        result = analyze_stock(
            "600519",
            perspective="value",
            depth="summary",
            factor_context={"composite_score": 0.9, "value_score": 0.8,
                            "quality_score": 0.85, "growth_score": 0.5, "roe": 28.3},
            pre_fetched=mock_endpoints,
            client=mock_llm_client,
        )
        for key in ("ticker", "name", "markdown", "fetched_endpoints",
                    "tokens_used", "generated_at"):
            assert key in result, f"missing key {key}"

    def test_markdown_has_disclaimer(self, mock_endpoints, mock_llm_client):
        result = analyze_stock(
            "600519",
            factor_context={"composite_score": 0.9},
            pre_fetched=mock_endpoints,
            client=mock_llm_client,
        )
        assert result["markdown"] is not None
        # 命门: 必须含「不构成投资建议」+「AI 生成」
        assert "不构成投资建议" in result["markdown"]
        assert "AI 生成" in result["markdown"]

    def test_tokens_counted(self, mock_endpoints, mock_llm_client):
        result = analyze_stock(
            "600519",
            pre_fetched=mock_endpoints,
            client=mock_llm_client,
        )
        assert result["tokens_used"] == 700  # 500 + 200

    def test_normalize_ticker(self, mock_endpoints, mock_llm_client):
        result = analyze_stock(
            "SH600519",
            pre_fetched=mock_endpoints,
            client=mock_llm_client,
        )
        assert result["ticker"] == "600519"

    def test_factor_context_in_prompt(self, mock_endpoints, mock_llm_client):
        """factor_context 字段 (综合分/估值分位等) 应进入 user prompt."""
        analyze_stock(
            "600519",
            factor_context={
                "composite_score": 0.95,
                "value_score": 0.88,
                "roe": 28.3,
            },
            pre_fetched=mock_endpoints,
            client=mock_llm_client,
        )
        call = mock_llm_client.chat.call_args
        user_msg = call.args[0][0]["content"]
        assert "0.95" in user_msg or "0.950" in user_msg  # composite_score
        assert "ROE" in user_msg

    def test_perspective_value_uses_value_system(self, mock_endpoints, mock_llm_client):
        analyze_stock(
            "600519",
            perspective="value",
            pre_fetched=mock_endpoints,
            client=mock_llm_client,
        )
        call = mock_llm_client.chat.call_args
        system = call.kwargs.get("system", "")
        assert "价值" in system

    def test_pre_fetched_skips_network(self, mock_endpoints, mock_llm_client):
        """pre_fetched 提供时不再调 SKILL adapter (这里通过 mock 整个 _fetch 验证)."""
        with patch("astock_quant.llm.stock_analyst._fetch_endpoints_for_value") as mock_fetch:
            analyze_stock(
                "600519",
                pre_fetched=mock_endpoints,
                client=mock_llm_client,
            )
            mock_fetch.assert_not_called()


# ===========================================================================
# analyze_stock 失败兜底
# ===========================================================================


class TestAnalyzeStockFallback:
    def test_llm_init_failure_returns_fallback(self, mock_endpoints):
        with patch("astock_quant.llm.stock_analyst._make_client",
                   side_effect=LLMClientError("no key")):
            result = analyze_stock("600519", pre_fetched=mock_endpoints)
        assert result["markdown"] is None
        assert "error" in result
        assert result["ticker"] == "600519"

    def test_llm_call_failure_returns_fallback(self, mock_endpoints):
        bad_client = MagicMock()
        bad_client.chat.side_effect = LLMClientError("API down")
        result = analyze_stock(
            "600519",
            pre_fetched=mock_endpoints,
            client=bad_client,
        )
        assert result["markdown"] is None
        assert "error" in result
        assert "API down" in result["error"]

    def test_empty_llm_response_returns_fallback(self, mock_endpoints):
        empty_client = MagicMock()
        empty_client.chat.return_value = LLMResponse(content="", usage=None, model="x")
        result = analyze_stock(
            "600519",
            pre_fetched=mock_endpoints,
            client=empty_client,
        )
        assert result["markdown"] is None


# ===========================================================================
# market_overview
# ===========================================================================


class TestMarketOverview:
    def test_returns_markdown_with_disclaimer(self):
        client = MagicMock()
        client.chat.return_value = LLMResponse(
            content="今日 AI / 算力题材最热, 北向资金净流入。",
            usage={"input_tokens": 200, "output_tokens": 100},
            model="x",
        )
        with patch("astock_quant.llm.stock_analyst._fetch_endpoints_for_market",
                   return_value={"hot": pd.DataFrame(), "north": None, "dragon": None}):
            result = market_overview(
                picks_summary=[{"ticker": "600519", "composite_score": 0.9}],
                client=client,
            )
        assert result["markdown"] is not None
        assert "不构成投资建议" in result["markdown"]
        assert "AI 生成" in result["markdown"]

    def test_llm_failure_returns_none_markdown(self):
        bad_client = MagicMock()
        bad_client.chat.side_effect = LLMClientError("boom")
        with patch("astock_quant.llm.stock_analyst._fetch_endpoints_for_market",
                   return_value={"hot": None, "north": None, "dragon": None}):
            result = market_overview(
                picks_summary=[{"ticker": "600519"}],
                client=bad_client,
            )
        assert result["markdown"] is None
        assert "error" in result


# ===========================================================================
# disclaimer 常量本身就有「不构成投资建议」
# ===========================================================================


class TestDisclaimer:
    def test_disclaimer_content(self):
        assert "不构成投资建议" in DISCLAIMER
        assert "AI 生成" in DISCLAIMER
