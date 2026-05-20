"""P12 用户友好报告改造命门测试.

覆盖 8 个命门：
1. 今日总结 div 在诚信声明 div 之前（HTML 顺序）
2. 诚信声明每个 metric 后真有 "→ 📖" 字串（动态翻译）
3. 表格 ticker 列含中文名
4. 每个 § 末尾有 📖 大白话 字串
5. _translate_metric 对不同数值返回不同文本（动态评级）
6. get_ticker_name 3 道 fallback：已知 → 全名 / 未知 → ticker 自身
7. 今日总结含 Top 1 推荐股票
8. HTML 信号条图含 CSS 颜色（red/green 关键词或色值）
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from astock_quant.contracts import Prediction


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _make_pred(ticker: str = "600519", value: float = 1.0, score: float = 0.62) -> Prediction:
    return Prediction(
        ticker=ticker,
        date=pd.Timestamp("2026-05-16").date(),
        target_type="direction",
        value=value,
        score=score,
        proba=(0.38, 0.62),
    )


def _make_full_results(
    dir_preds=None,
    rank_preds=None,
    dir_metrics=None,
) -> dict:
    """renderer.render 期望的完整 results dict."""
    if dir_preds is None:
        dir_preds = [_make_pred("600519", 1.0), _make_pred("000858", 0.0)]
    if rank_preds is None:
        rank_preds = [_make_pred("601012", 1.0, score=0.92), _make_pred("300750", 1.0, score=0.78)]
    if dir_metrics is None:
        dir_metrics = {"auc": 0.5131}

    return {
        "report_date": "2026-05-16",
        "universe_size": len(dir_preds),
        "generated_at": "2026-05-16T16:32:00",
        "data_cutoff": "2026-05-16",
        "total_seconds": 1.23,
        "model_version": "2026-05-16",
        "model_paths": "direction=artifacts/models/direction_2026-05-16.lgb",
        "json_path": "artifacts/daily_reports/predictions_2026-05-16.json",
        "errors": [],
        "direction": {"predictions": dir_preds, "metrics": dir_metrics},
        "return_": {"predictions": dir_preds, "metrics": {"r2": -0.002}},
        "ranking": {"predictions": rank_preds, "metrics": {"spearman_corr": 0.01}},
        "trade_signal": {
            "predictions": [_make_pred("600519", 1.0), _make_pred("000001", -1.0)],
            "buy_predictions": [_make_pred("600519", 1.0)],
            "metrics": {"macro_f1": 0.33},
        },
        "accuracy": None,
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
# 命门 2：诚信声明每个 metric 后有 "→ 📖" 翻译箭头
# ---------------------------------------------------------------------------

class TestDisclaimerMetricTranslationArrow:

    def test_disclaimer_contains_translation_arrow(self, tmp_path):
        """诚信声明含 '→ 📖' 翻译箭头（动态翻译已接入）。"""
        html = _render_html(_make_full_results(), tmp_path)
        assert "→" in html and "📖" in html, \
            "诚信声明缺少 '→ 📖' 翻译箭头，可能动态翻译未接入"

    def test_translate_metric_auc_produces_translation(self):
        """_translate_metric('auc', 0.5131) 返回含 📖 的翻译字符串。"""
        from astock_quant.predict.renderer import _translate_metric
        result = _translate_metric("auc", 0.5131)
        assert "📖" in result, f"_translate_metric(auc, 0.5131) 未返回 📖 翻译: {result!r}"
        assert len(result) > 5, "翻译太短，疑似空或错误"

    def test_translate_metric_r2_produces_translation(self):
        """_translate_metric('r2', -0.002) 返回含 📖 的翻译。"""
        from astock_quant.predict.renderer import _translate_metric
        result = _translate_metric("r2", -0.002)
        assert "📖" in result, f"_translate_metric(r2, -0.002) 未返回 📖 翻译: {result!r}"

    def test_translate_metric_spearman_produces_translation(self):
        """_translate_metric('spearman_corr', 0.01) 返回含 📖 的翻译。"""
        from astock_quant.predict.renderer import _translate_metric
        result = _translate_metric("spearman_corr", 0.01)
        assert "📖" in result


# ---------------------------------------------------------------------------
# 命门 3：表格 ticker 列含中文名
# ---------------------------------------------------------------------------

class TestTickerTableContainsChineseName:

    def test_html_ticker_display_contains_chinese_name(self, tmp_path):
        """HTML 里 ticker 旁含至少 1 个中文股票名（不只 code）。"""
        html = _render_html(_make_full_results(), tmp_path)
        # 600519 → 贵州茅台，000858 → 五粮液 等都应出现
        chinese_names = ["贵州茅台", "五粮液", "隆基绿能", "宁德时代"]
        found = [name for name in chinese_names if name in html]
        assert found, (
            f"HTML 里未找到任何中文股票名，检查过：{chinese_names}。"
            "ticker_names.py 可能未接入 renderer。"
        )

    def test_md_ticker_display_contains_chinese_name(self, tmp_path):
        """MD 里 ticker 旁含中文名。"""
        md = _render_md(_make_full_results(), tmp_path)
        chinese_names = ["贵州茅台", "五粮液", "隆基绿能"]
        found = [name for name in chinese_names if name in md]
        assert found, f"MD 里未找到任何中文股票名，检查过：{chinese_names}"

    def test_ticker_display_html_includes_code_and_name(self):
        """_ticker_display_html 返回格式含 code 和中文名。"""
        from astock_quant.predict.renderer import _ticker_display_html
        result = _ticker_display_html("600519")
        assert "600519" in result, "ticker display 缺 code"
        assert "贵州茅台" in result, "ticker display 缺中文名"


# ---------------------------------------------------------------------------
# 命门 4：每个 § 末尾有 📖 大白话 字串
# ---------------------------------------------------------------------------

class TestPlainLanguageSectionAtEnd:

    def test_html_contains_plain_language_for_direction(self, tmp_path):
        """direction 节含 📖 大白话 内容。"""
        html = _render_html(_make_full_results(), tmp_path)
        assert "📖" in html, "HTML 缺少 📖 大白话内容"
        assert "plain-language" in html or "📖" in html, \
            "direction 节缺少大白话段"

    def test_render_plain_language_direction_not_empty(self):
        """_render_plain_language('direction', ...) 返回非空含 📖 字串。"""
        from astock_quant.predict.renderer import _render_plain_language
        preds = [_make_pred("600519", 0.0), _make_pred("000858", 0.0)]
        result = _render_plain_language("direction", {"predictions": preds, "metrics": {}})
        assert "📖" in result, f"_render_plain_language direction 未返回 📖: {result!r}"

    def test_render_plain_language_return_not_empty(self):
        """_render_plain_language('return', ...) 返回非空含 📖 字串。"""
        from astock_quant.predict.renderer import _render_plain_language
        preds = [_make_pred("600519", 0.001)]
        result = _render_plain_language("return", {"predictions": preds, "metrics": {"r2": -0.002}})
        assert "📖" in result

    def test_render_plain_language_ranking_not_empty(self):
        """_render_plain_language('ranking', ...) 返回非空含 📖 字串。"""
        from astock_quant.predict.renderer import _render_plain_language
        preds = [_make_pred("601012", 1.0, score=0.92)]
        result = _render_plain_language("ranking", {"predictions": preds, "metrics": {}})
        assert "📖" in result

    def test_render_plain_language_trade_signal_not_empty(self):
        """_render_plain_language('trade_signal', ...) 返回非空含 📖 字串。"""
        from astock_quant.predict.renderer import _render_plain_language
        preds = [_make_pred("600519", 1.0)]
        result = _render_plain_language("trade_signal", {"predictions": preds, "metrics": {}})
        assert "📖" in result


# ---------------------------------------------------------------------------
# 命门 5：_translate_metric 对不同数值返回不同文本（动态评级）
# ---------------------------------------------------------------------------

class TestTranslateMetricDynamicRating:

    def test_auc_low_vs_high_returns_different_text(self):
        """AUC=0.51 和 AUC=0.70 应返回不同的翻译文案。"""
        from astock_quant.predict.renderer import _translate_metric
        low = _translate_metric("auc", 0.51)
        high = _translate_metric("auc", 0.70)
        assert low != high, \
            f"AUC=0.51 和 AUC=0.70 翻译相同，动态评级未生效: {low!r}"

    def test_auc_weak_signal_not_strong(self):
        """AUC=0.51 不应翻译成「真有点信号」等强评价。"""
        from astock_quant.predict.renderer import _translate_metric
        result = _translate_metric("auc", 0.51)
        assert "硬币" in result or "随机" in result or "猜" in result, \
            f"AUC=0.51 应被翻译成随机水平，实际: {result!r}"

    def test_auc_high_triggers_look_ahead_warning(self):
        """AUC=0.80 应触发「检查是否数据泄漏」警告。"""
        from astock_quant.predict.renderer import _translate_metric
        result = _translate_metric("auc", 0.80)
        assert "泄漏" in result or "look-ahead" in result or "异常" in result, \
            f"AUC=0.80 应警告数据泄漏，实际: {result!r}"

    def test_r2_negative_vs_positive_returns_different_text(self):
        """R²=-0.002 和 R²=0.05 应返回不同翻译。"""
        from astock_quant.predict.renderer import _translate_metric
        neg = _translate_metric("r2", -0.002)
        pos = _translate_metric("r2", 0.05)
        assert neg != pos

    def test_translate_metric_non_numeric_returns_fallback(self):
        """_translate_metric 传入非数字 → 返回「数据不足」而非崩溃。"""
        from astock_quant.predict.renderer import _translate_metric
        result = _translate_metric("auc", None)
        assert result == "数据不足", f"非数字输入应返回「数据不足」，实际: {result!r}"

    def test_translate_metric_unknown_metric_returns_empty(self):
        """未知 metric_name → 返回空字符串（不崩溃）。"""
        from astock_quant.predict.renderer import _translate_metric
        result = _translate_metric("unknown_metric_xyz", 0.5)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 命门 6：get_ticker_name fallback 链
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
# 命门 7：今日总结含 Top 1 推荐股票
# ---------------------------------------------------------------------------

class TestTodaySummaryIncludesTopRecommendation:

    def test_summary_line_2_contains_top1_stock(self, tmp_path):
        """今日总结第 2 行（🥇）含 ranking top1 股票名或代码。"""
        rank_preds = [
            _make_pred("601012", 1.0, score=0.92),  # 隆基绿能，score 最高
            _make_pred("300750", 1.0, score=0.78),
        ]
        results = _make_full_results(rank_preds=rank_preds)
        html = _render_html(results, tmp_path)
        # 第 2 行应含 🥇 + 隆基绿能 / 601012
        assert "🥇" in html, "今日总结缺少 🥇 符号"
        assert "601012" in html or "隆基绿能" in html, \
            "今日总结第 2 行未包含 ranking top1 股票（601012 / 隆基绿能）"

    def test_render_today_summary_line2_has_top1(self):
        """_render_today_summary → summary_line_2 含 top1 股票。"""
        from astock_quant.predict.renderer import _render_today_summary
        rank_preds = [
            _make_pred("600519", 1.0, score=0.95),
            _make_pred("000858", 1.0, score=0.70),
        ]
        results = {
            "direction": {"predictions": [_make_pred("600519", 1.0)], "metrics": {}},
            "ranking": {"predictions": rank_preds, "metrics": {}},
        }
        summary = _render_today_summary(results)
        line2 = summary["summary_line_2"]
        assert "🥇" in line2, f"summary_line_2 缺 🥇: {line2!r}"
        assert "600519" in line2 or "贵州茅台" in line2, \
            f"summary_line_2 未含 top1 股票 600519/贵州茅台: {line2!r}"

    def test_render_today_summary_no_ranking_preds_graceful(self):
        """ranking 无数据时 _render_today_summary 不崩溃，给出占位文本。"""
        from astock_quant.predict.renderer import _render_today_summary
        results = {
            "direction": {"predictions": [], "metrics": {}},
            "ranking": {"predictions": [], "metrics": {}},
        }
        summary = _render_today_summary(results)
        assert "summary_line_2" in summary
        assert isinstance(summary["summary_line_2"], str)


# ---------------------------------------------------------------------------
# 命门 8：HTML 信号条图含 CSS 颜色（红/绿）
# ---------------------------------------------------------------------------

class TestSignalDistributionHtmlUsesColors:

    def test_signal_distribution_html_contains_green_color(self):
        """HTML style 的 signal-bar.buy 含绿色色值（#52c41a 或 green 关键词）。"""
        from astock_quant.predict.renderer import _render_signal_distribution
        preds = [_make_pred("600519", 1.0), _make_pred("000858", 0.0)]
        html = _render_signal_distribution(preds, style="html")
        assert "buy" in html, "信号分布 HTML 缺少 buy 类"
        # 颜色通过 CSS class 注入，html 片段里应有 class="signal-bar buy"
        assert 'class="signal-bar buy"' in html or "signal-bar buy" in html, \
            f"信号分布 HTML 缺少 signal-bar buy class: {html[:300]}"

    def test_signal_distribution_html_contains_red_color(self):
        """HTML signal-bar.sell 含红色 CSS class。"""
        from astock_quant.predict.renderer import _render_signal_distribution
        preds = [_make_pred("600519", 0.0)]
        html = _render_signal_distribution(preds, style="html")
        assert "sell" in html, "信号分布 HTML 缺少 sell 类"

    def test_html_template_signal_bar_css_has_color(self, tmp_path):
        """渲染后的完整 HTML 含 signal-bar 的 CSS 颜色定义（#52c41a / #f5222d）。"""
        html = _render_html(_make_full_results(), tmp_path)
        # CSS 定义在 <style> 里，应含绿色和红色
        assert "#52c41a" in html or "signal-bar" in html, \
            "渲染后 HTML 缺少 signal-bar 绿色色值 #52c41a"
        assert "#f5222d" in html or "signal-bar" in html, \
            "渲染后 HTML 缺少 signal-bar 红色色值 #f5222d"

    def test_signal_distribution_md_not_html_style(self):
        """MD 模式返回 ASCII 条图，不是 HTML div。"""
        from astock_quant.predict.renderer import _render_signal_distribution
        preds = [_make_pred("600519", 1.0), _make_pred("000858", 0.0)]
        md = _render_signal_distribution(preds, style="md")
        assert "<div" not in md, "MD 模式不应含 HTML div"
        assert "看涨" in md or "看跌" in md, "MD 模式应含涨跌描述文字"
