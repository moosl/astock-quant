"""用户友好报告命门测试 —— renderer.py 价值选股版.

覆盖：
- 今日总结 div 在诚信声明 div 之前（HTML 顺序）
- 今日总结含本期综合分第一的价值股
- §1 价值名单表格 ticker 列含中文名
- get_ticker_name 3 道 fallback：已知 → 全名 / 未知 → ticker 自身

注：2026-05-22 项目「涨跌预测 → 价值选股」改造后，旧的 4 个短期模型
（direction/return/ranking/trade_signal）服务函数 make_ascii_bar /
_translate_metric / _render_signal_distribution / _render_plain_language
已从 renderer.py 删除，对应单测一并移除。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _make_full_results() -> dict:
    """renderer.render 期望的 results dict（价值选股版）."""
    return {
        "report_date": "2026-05-16",
        "universe_size": 2,
        "generated_at": "2026-05-16T16:32:00",
        "data_cutoff": "2026-05-16",
        "total_seconds": 1.23,
        "model_version": "2026-05-16",
        "json_path": "artifacts/daily_reports/predictions_2026-05-16.json",
        "errors": [],
        "accuracy": None,
        "value_picks": None,
        "backtest": None,
    }


def _render_html(results: dict, tmp_path: Path) -> str:
    from astock_quant.predict.renderer import render
    html_path, _ = render(results, tmp_path)
    return html_path.read_text(encoding="utf-8")


def _render_md(results: dict, tmp_path: Path) -> str:
    from astock_quant.predict.renderer import render
    _, md_path = render(results, tmp_path)
    return md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 命门 1：today-summary div 在 disclaimer div 之前
# ---------------------------------------------------------------------------

class TestHtmlTodaySummaryBeforeDisclaimer:

    def test_today_summary_div_appears_before_disclaimer_div(self, tmp_path):
        """today-summary 的位置必须在 disclaimer 之前（顺序守门）。"""
        html = _render_html(_make_full_results(), tmp_path)
        idx_summary = html.find('class="today-summary"')
        idx_disclaimer = html.find('class="disclaimer"')
        assert idx_summary != -1, "HTML 缺少 today-summary div"
        assert idx_disclaimer != -1, "HTML 缺少 disclaimer div"
        assert idx_summary < idx_disclaimer, (
            f"today-summary (pos={idx_summary}) 应在 disclaimer (pos={idx_disclaimer}) 之前，"
            "实际顺序相反"
        )

    def test_today_summary_has_all_3_lines(self, tmp_path):
        """today-summary 含 3 行 summary-line（第 3 行 class 含 honesty）。"""
        html = _render_html(_make_full_results(), tmp_path)
        # line3 is class="summary-line honesty" — count substring "summary-line"
        total = html.count("summary-line")
        assert total >= 3, \
            f"today-summary 应有 3 个 summary-line 元素，实际 {total}"

    def test_today_summary_contains_today_one_liner_emoji(self, tmp_path):
        """today-summary 里有 🎯 emoji（第 1 行标志）。"""
        html = _render_html(_make_full_results(), tmp_path)
        assert "🎯" in html, "today-summary 缺少 🎯 一句话总结标志"


# ---------------------------------------------------------------------------
# 命门 2：表格 ticker 列含中文名
# ---------------------------------------------------------------------------

def _results_with_value_picks() -> dict:
    """带 value_picks 的 results —— §1 价值名单是报告里展示 ticker 中文名的地方。"""
    results = _make_full_results()
    results["value_picks"] = [
        {"ticker": "600519", "composite_score": 0.85, "pe": 26.0, "pb": 8.0,
         "roe": 28.0, "reason": "盈利质量强"},
        {"ticker": "000858", "composite_score": 0.72, "pe": 20.0, "pb": 5.0,
         "roe": 22.0, "reason": "估值合理"},
    ]
    return results


class TestTickerTableContainsChineseName:

    def test_html_ticker_display_contains_chinese_name(self, tmp_path):
        """HTML 里 ticker 旁含至少 1 个中文股票名（§1 价值名单接入 ticker_names）。"""
        html = _render_html(_results_with_value_picks(), tmp_path)
        # 600519 → 贵州茅台，000858 → 五粮液
        chinese_names = ["贵州茅台", "五粮液", "隆基绿能", "宁德时代"]
        found = [name for name in chinese_names if name in html]
        assert found, (
            f"HTML 里未找到任何中文股票名，检查过：{chinese_names}。"
            "ticker_names.py 可能未接入 renderer。"
        )

    def test_md_ticker_display_contains_chinese_name(self, tmp_path):
        """MD 里 ticker 旁含中文名（§1 价值名单）。"""
        md = _render_md(_results_with_value_picks(), tmp_path)
        chinese_names = ["贵州茅台", "五粮液", "隆基绿能"]
        found = [name for name in chinese_names if name in md]
        assert found, f"MD 里未找到任何中文股票名，检查过：{chinese_names}"


# ---------------------------------------------------------------------------
# 命门 3：get_ticker_name fallback 链
# ---------------------------------------------------------------------------

class TestGetTickerNameFallbackChain:

    def test_known_ticker_returns_chinese_name(self):
        """600519 → 贵州茅台（硬编码 fallback 第 1 道）。"""
        from astock_quant.predict.ticker_names import get_ticker_name
        assert get_ticker_name("600519") == "贵州茅台"

    def test_known_ticker_601012_returns_correct_name(self):
        """601012 → 隆基绿能。"""
        from astock_quant.predict.ticker_names import get_ticker_name
        assert get_ticker_name("601012") == "隆基绿能"

    def test_known_ticker_601318_returns_correct_name(self):
        """601318 → 中国平安（确保茅台/隆基/平安等核心名映射正确）。"""
        from astock_quant.predict.ticker_names import get_ticker_name
        assert get_ticker_name("601318") == "中国平安"

    def test_unknown_ticker_returns_code_itself(self):
        """未知 ticker → 兜底返回 code 自身，不挂。"""
        from astock_quant.predict.ticker_names import get_ticker_name

        with (
            patch("astock_quant.predict.ticker_names._load_cache", return_value={}),
            patch("astock_quant.predict.ticker_names._fetch_from_akshare",
                  side_effect=Exception("网络超时")),
        ):
            result = get_ticker_name("999999")
        assert result == "999999", f"未知 ticker 应兜底返回 code 本身，实际: {result!r}"

    def test_cache_hit_returns_cached_name(self):
        """cache 命中 → 返回 cache 里的名字（不调 akshare）。"""
        from astock_quant.predict.ticker_names import get_ticker_name

        mock_ak = MagicMock()
        with (
            patch("astock_quant.predict.ticker_names._load_cache",
                  return_value={"999001": "测试股份"}),
            patch("astock_quant.predict.ticker_names._fetch_from_akshare", mock_ak),
        ):
            result = get_ticker_name("999001")

        assert result == "测试股份"
        mock_ak.assert_not_called()

    def test_all_30_stage1_tickers_have_names(self):
        """STAGE1_NAMES 恰好 30 只，每个 value 非空。"""
        from astock_quant.predict.ticker_names import STAGE1_NAMES
        assert len(STAGE1_NAMES) == 30, f"STAGE1_NAMES 应有 30 只，实际 {len(STAGE1_NAMES)}"
        for code, name in STAGE1_NAMES.items():
            assert name, f"code={code} 的中文名为空"
            assert code.isdigit() and len(code) == 6, f"code={code!r} 不是 6 位纯数字"


# ---------------------------------------------------------------------------
# 命门 4：今日速览围绕价值选股名单（不再读旧涨跌预测模型）
#
# 注：2026-05-22 移除旧涨跌预测章节后，今日速览改为讲「本期综合分第一的价值股」，
# 不再讲「ranking top1」「① 强势评分」。
# ---------------------------------------------------------------------------

def _make_value_picks_for_summary() -> list[dict]:
    return [
        {"ticker": "601838", "composite_score": 0.799, "pe": 6.0, "pb": 0.9,
         "roe": 15.2, "reason": "估值低、ROE 高"},
        {"ticker": "600519", "composite_score": 0.72, "pe": 26.0, "pb": 8.0,
         "roe": 28.0, "reason": "盈利质量强"},
    ]


class TestTodaySummaryIncludesTopRecommendation:

    def test_summary_line_1_contains_top_value_pick(self, tmp_path):
        """今日速览第 1 行含本期综合分第一的价值股代码。"""
        results = _make_full_results()
        results["value_picks"] = _make_value_picks_for_summary()
        html = _render_html(results, tmp_path)
        assert "🎯" in html, "今日速览缺少 🎯 符号"
        assert "601838" in html, "今日速览未包含本期综合分第一的票 601838"

    def test_render_today_summary_line1_has_top_value_pick(self):
        """_render_today_summary → summary_line_1 含综合分最高的价值股。"""
        from astock_quant.predict.renderer import _render_today_summary
        results = {"value_picks": _make_value_picks_for_summary(), "backtest": {}}
        summary = _render_today_summary(results)
        line1 = summary["summary_line_1"]
        assert "🎯" in line1, f"summary_line_1 缺 🎯: {line1!r}"
        assert "601838" in line1 or "成都银行" in line1, \
            f"summary_line_1 未含综合分第一的票 601838/成都银行: {line1!r}"

    def test_render_today_summary_no_old_prediction_traces(self):
        """命门：今日速览不再出现旧涨跌预测痕迹（强势评分 / AUC / ranking）。"""
        from astock_quant.predict.renderer import _render_today_summary
        results = {"value_picks": _make_value_picks_for_summary(), "backtest": {}}
        summary = _render_today_summary(results)
        joined = " ".join(summary.values())
        assert "强势评分" not in joined, f"今日速览残留「强势评分」: {joined!r}"
        assert "AUC" not in joined, f"今日速览残留「AUC」: {joined!r}"

    def test_render_today_summary_no_value_picks_graceful(self):
        """value_picks 无数据时 _render_today_summary 不崩溃，给出占位文本。"""
        from astock_quant.predict.renderer import _render_today_summary
        results = {"value_picks": [], "backtest": {}}
        summary = _render_today_summary(results)
        for key in ("summary_line_1", "summary_line_2", "summary_line_3"):
            assert key in summary
            assert isinstance(summary[key], str)
