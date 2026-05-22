"""每日价值选股报告测试 —— daily.py + renderer.py.

覆盖：
- _resolve_date：'today' / ISO / 非法格式
- run_daily_predict：JSON 运行记录落盘 schema / renderer 失败时 exit 1 / --no-render
- CLI main()：exit code 路径 / universe 解析
- renderer.render：HTML + MD 双输出 / 命门诚信声明守门
- _build_value_picks / _try_build_value_picks：价值选股名单组装
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_full_results(date_str: str = "2026-05-16") -> dict:
    """renderer.render 期望的 results dict（价值选股版）."""
    return {
        "report_date": date_str,
        "universe_size": 2,
        "generated_at": f"{date_str}T16:32:00",
        "data_cutoff": date_str,
        "total_seconds": 1.23,
        "model_version": date_str,
        "json_path": f"artifacts/daily_reports/predictions_{date_str}.json",
        "errors": [],
        "accuracy": None,
        "value_picks": None,
        "backtest": None,
    }


# ---------------------------------------------------------------------------
# _resolve_date
# ---------------------------------------------------------------------------

class TestResolveDate:

    def test_today_resolves_to_iso(self):
        from astock_quant.predict.daily import _resolve_date
        import datetime
        result = _resolve_date("today")
        assert result == datetime.date.today().isoformat()

    def test_none_resolves_to_today(self):
        from astock_quant.predict.daily import _resolve_date
        import datetime
        result = _resolve_date(None)
        assert result == datetime.date.today().isoformat()

    def test_iso_date_passthrough(self):
        from astock_quant.predict.daily import _resolve_date
        assert _resolve_date("2026-05-15") == "2026-05-15"

    def test_invalid_date_raises(self):
        from astock_quant.predict.daily import _resolve_date
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            _resolve_date("2026/05/15")

    def test_invalid_date_string_raises(self):
        from astock_quant.predict.daily import _resolve_date
        with pytest.raises(ValueError):
            _resolve_date("not-a-date")


# ---------------------------------------------------------------------------
# run_daily_predict — 核心流程
# ---------------------------------------------------------------------------

class TestRunDailyPredict:

    def test_json_payload_schema(self, tmp_path):
        """JSON 运行记录落盘后含 report_date / universe_size / errors / value_picks keys."""
        with patch("astock_quant.predict.daily._try_build_value_picks", return_value=None), \
             patch("astock_quant.predict.daily._load_backtest_for_report", return_value=None):
            from astock_quant.predict.daily import run_daily_predict
            run_daily_predict(
                date="2026-05-16",
                universe=["600519", "000858"],
                output_dir=tmp_path,
                render_report=False,
            )

        json_file = tmp_path / "predictions_2026-05-16.json"
        assert json_file.exists(), "JSON 文件未落盘"
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        for key in ["report_date", "universe_size", "errors", "value_picks"]:
            assert key in payload, f"JSON payload 缺 key: {key}"
        assert payload["report_date"] == "2026-05-16"
        assert payload["universe_size"] == 2

    def test_json_payload_has_no_old_target_results(self, tmp_path):
        """JSON 运行记录不再含旧的 4-target results 段（涨跌预测残留已清）."""
        with patch("astock_quant.predict.daily._try_build_value_picks", return_value=None), \
             patch("astock_quant.predict.daily._load_backtest_for_report", return_value=None):
            from astock_quant.predict.daily import run_daily_predict
            run_daily_predict(
                date="2026-05-16",
                universe=["600519"],
                output_dir=tmp_path,
                render_report=False,
            )

        payload = json.loads((tmp_path / "predictions_2026-05-16.json").read_text())
        assert "results" not in payload, "JSON 仍残留旧的 4-target results 段"
        assert "model_paths" not in payload, "JSON 仍残留旧的 model_paths 字段"

    def test_value_picks_in_json_payload(self, tmp_path):
        """value_picks 算出来后会写进 JSON 运行记录。"""
        fake_picks = [{"ticker": "600519", "composite_score": 0.9, "reason": "test"}]
        with patch("astock_quant.predict.daily._try_build_value_picks", return_value=fake_picks), \
             patch("astock_quant.predict.daily._load_backtest_for_report", return_value=None):
            from astock_quant.predict.daily import run_daily_predict
            results = run_daily_predict(
                date="2026-05-16",
                universe=["600519"],
                output_dir=tmp_path,
                render_report=False,
            )

        assert results["value_picks"] == fake_picks
        payload = json.loads((tmp_path / "predictions_2026-05-16.json").read_text())
        assert payload["value_picks"] == fake_picks

    def test_renderer_failure_recorded_in_errors(self, tmp_path):
        """renderer 抛异常时错误记进 errors + error log 落盘。"""
        with patch("astock_quant.predict.daily._try_build_value_picks", return_value=None), \
             patch("astock_quant.predict.daily._load_backtest_for_report", return_value=None), \
             patch("astock_quant.predict.renderer.render", side_effect=RuntimeError("渲染爆了")):
            from astock_quant.predict.daily import run_daily_predict
            results = run_daily_predict(
                date="2026-05-16",
                universe=["600519"],
                output_dir=tmp_path,
                render_report=True,
            )

        assert len(results["errors"]) >= 1
        assert any("渲染爆了" in e for e in results["errors"])
        log_file = tmp_path / "error_2026-05-16.log"
        assert log_file.exists(), "renderer 失败时应生成 error log"

    def test_no_error_log_when_clean_run(self, tmp_path):
        """无错误 → 不生成 error log。"""
        with patch("astock_quant.predict.daily._try_build_value_picks", return_value=None), \
             patch("astock_quant.predict.daily._load_backtest_for_report", return_value=None):
            from astock_quant.predict.daily import run_daily_predict
            run_daily_predict(
                date="2026-05-16",
                universe=["600519"],
                output_dir=tmp_path,
                render_report=False,
            )

        log_file = tmp_path / "error_2026-05-16.log"
        assert not log_file.exists(), "无错误时不应生成 error log"

    def test_no_render_skips_html_md(self, tmp_path):
        """render_report=False 不生成 HTML/MD 文件。"""
        with patch("astock_quant.predict.daily._try_build_value_picks", return_value=None), \
             patch("astock_quant.predict.daily._load_backtest_for_report", return_value=None):
            from astock_quant.predict.daily import run_daily_predict
            results = run_daily_predict(
                date="2026-05-16",
                universe=["600519"],
                output_dir=tmp_path,
                render_report=False,
            )

        assert "html_path" not in results
        assert "md_path" not in results
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) == 0, f"--no-render 下不应有 HTML: {html_files}"

    def test_prepare_stage1_data_called_once(self, tmp_path):
        """prepare_stage1_data 在一次运行里只调 1 次（数据共享，不重拉）。"""
        prepare_call_count = [0]
        fake_data = {"prices": object(), "moneyflow": None, "financials": {}}

        def mock_prepare(universe=None, force_refresh=False):
            prepare_call_count[0] += 1
            return fake_data

        with patch("astock_quant.data.dataset.prepare_stage1_data", side_effect=mock_prepare), \
             patch("astock_quant.predict.daily._try_build_value_picks", return_value=None), \
             patch("astock_quant.predict.daily._load_backtest_for_report", return_value=None):
            from astock_quant.predict.daily import run_daily_predict
            run_daily_predict(
                date="2026-05-16",
                universe=["600519", "000858"],
                output_dir=tmp_path,
                render_report=False,
            )

        assert prepare_call_count[0] <= 1, \
            f"prepare_stage1_data 被调了 {prepare_call_count[0]} 次，应该 <= 1 次"


# ---------------------------------------------------------------------------
# CLI main() exit codes
# ---------------------------------------------------------------------------

class TestDailyMain:

    def test_main_clean_run_exit_0(self, tmp_path):
        """正常运行 → exit 0。"""
        with patch("astock_quant.predict.daily._try_build_value_picks", return_value=None), \
             patch("astock_quant.predict.daily._load_backtest_for_report", return_value=None):
            from astock_quant.predict.daily import main
            exit_code = main([
                "--date", "2026-05-16",
                "--universe", "600519,000858",
                "--output-dir", str(tmp_path),
                "--no-render",
            ])
        assert exit_code == 0

    def test_main_renderer_failure_exit_1(self, tmp_path):
        """renderer 失败 → exit 1。"""
        with patch("astock_quant.predict.daily._try_build_value_picks", return_value=None), \
             patch("astock_quant.predict.daily._load_backtest_for_report", return_value=None), \
             patch("astock_quant.predict.renderer.render", side_effect=RuntimeError("渲染爆了")):
            from astock_quant.predict.daily import main
            exit_code = main([
                "--date", "2026-05-16",
                "--universe", "600519",
                "--output-dir", str(tmp_path),
            ])
        assert exit_code == 1

    def test_main_empty_universe_exit_2(self, tmp_path):
        """--universe 解析为空 → exit 2。"""
        from astock_quant.predict.daily import main
        exit_code = main([
            "--date", "2026-05-16",
            "--output-dir", str(tmp_path),
            "--universe", ",,,",
            "--no-render",
        ])
        assert exit_code == 2

    def test_main_universe_parsed_correctly(self, tmp_path):
        """--universe '600519,000858' 正确解析并传给 run_daily_predict。"""
        received = {}

        def mock_run(*, date, universe, output_dir, render_report):
            received["universe"] = list(universe) if universe else None
            return {"errors": []}

        with patch("astock_quant.predict.daily.run_daily_predict", side_effect=mock_run):
            from astock_quant.predict.daily import main
            main([
                "--date", "2026-05-16",
                "--universe", "600519,000858",
                "--output-dir", str(tmp_path),
                "--no-render",
            ])

        assert received.get("universe") is not None, "run_daily_predict 未收到 universe"
        assert set(received["universe"]) == {"600519", "000858"}, \
            f"universe 解析错误: {received['universe']}"

    def test_main_universe_stage4_returns_300_tickers(self, tmp_path):
        """--universe stage4 真返回沪深 300 universe（smoke test）。"""
        received = {}

        def mock_run(*, date, universe, output_dir, render_report):
            received["universe"] = list(universe) if universe else None
            return {"errors": []}

        with patch("astock_quant.predict.daily.run_daily_predict", side_effect=mock_run):
            from astock_quant.predict.daily import main
            exit_code = main([
                "--date", "2026-05-16",
                "--universe", "stage4",
                "--output-dir", str(tmp_path),
                "--no-render",
            ])

        assert exit_code == 0, f"--universe stage4 exit code 非 0: {exit_code}"
        assert received.get("universe") is not None
        assert len(received["universe"]) >= 200, \
            f"stage4 universe 只有 {len(received['universe'])} 只，期望 >= 200"

    def test_main_universe_hs300_alias_same_as_stage4(self, tmp_path):
        """--universe hs300 与 stage4 等价，返回同样大小的 universe。"""
        sizes = []

        def mock_run(*, date, universe, output_dir, render_report):
            sizes.append(len(universe) if universe else 0)
            return {"errors": []}

        with patch("astock_quant.predict.daily.run_daily_predict", side_effect=mock_run):
            from astock_quant.predict.daily import main
            main(["--date", "2026-05-16", "--universe", "stage4",
                  "--output-dir", str(tmp_path), "--no-render"])
            main(["--date", "2026-05-16", "--universe", "hs300",
                  "--output-dir", str(tmp_path), "--no-render"])

        assert len(sizes) == 2 and sizes[0] == sizes[1], \
            f"stage4({sizes[0]}) 与 hs300({sizes[1]}) universe 大小不一致"

    def test_main_universe_stage1_returns_30_tickers(self, tmp_path):
        """--universe stage1 返回 30 只 universe。"""
        received = {}

        def mock_run(*, date, universe, output_dir, render_report):
            received["universe"] = list(universe) if universe else None
            return {"errors": []}

        with patch("astock_quant.predict.daily.run_daily_predict", side_effect=mock_run):
            from astock_quant.predict.daily import main
            exit_code = main([
                "--date", "2026-05-16",
                "--universe", "stage1",
                "--output-dir", str(tmp_path),
                "--no-render",
            ])

        assert exit_code == 0
        assert len(received["universe"]) == 30, \
            f"stage1 universe 应为 30 只，实际: {len(received['universe'])}"


# ---------------------------------------------------------------------------
# renderer.render — HTML + MD 双输出
# ---------------------------------------------------------------------------

class TestRenderer:

    def test_render_creates_html_and_md(self, tmp_path):
        """render() 产生 HTML 和 MD 两个文件。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        html_path, md_path = render(results, tmp_path)
        assert html_path.exists(), "HTML 文件未生成"
        assert md_path.exists(), "MD 文件未生成"

    def test_render_html_contains_report_date(self, tmp_path):
        """HTML 里含报告日期。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results("2026-05-16")
        html_path, _ = render(results, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        assert "2026-05-16" in content

    def test_render_md_contains_report_date(self, tmp_path):
        """MD 里含报告日期。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results("2026-05-16")
        _, md_path = render(results, tmp_path)
        content = md_path.read_text(encoding="utf-8")
        assert "2026-05-16" in content

    def test_render_html_is_valid_html(self, tmp_path):
        """HTML 文件含 <html> 标签（最基本结构检查）。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        html_path, _ = render(results, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        assert "<html" in content.lower() or "<!doctype" in content.lower(), \
            "HTML 文件不含 <html> 标签"

    # -----------------------------------------------------------------------
    # 命门：诚信声明守门
    # -----------------------------------------------------------------------

    def test_daily_report_html_contains_honesty_disclaimer(self, tmp_path):
        """命门：HTML 报告必须含诚信声明关键词。防止有人删掉诚信声明。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        html_path, _ = render(results, tmp_path)
        content = html_path.read_text(encoding="utf-8")

        has_disclaimer = (
            "诚信声明" in content
            or "AUC" in content
            or "不构成投资" in content
            or "随机" in content
        )
        if not has_disclaimer:
            pytest.fail(
                "命门失败：HTML 报告缺少诚信声明。\n"
                "诚信声明是工程红线，必须出现在报告里。\n"
                "请检查 daily_report.html.template 是否还有 诚信声明 / 不构成投资 等关键词。"
            )

    def test_daily_report_md_contains_honesty_disclaimer(self, tmp_path):
        """命门：MD 报告必须含诚信声明关键词。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        _, md_path = render(results, tmp_path)
        content = md_path.read_text(encoding="utf-8")

        has_disclaimer = (
            "诚信声明" in content
            or "AUC" in content
            or "不构成投资" in content
            or "随机" in content
        )
        if not has_disclaimer:
            pytest.fail(
                "命门失败：MD 报告缺少诚信声明。\n"
                "请检查 daily_report.md.template 是否还有 诚信声明 等关键词。"
            )

    def test_template_placeholders_all_substituted(self, tmp_path):
        """渲染后 HTML 里不应残留任何 $xxx 模板占位符（全部已替换）。"""
        import re
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["value_picks"] = _make_value_picks()
        html_path, _ = render(results, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        leftover = re.findall(r"\$[a-z_]+", content)
        assert not leftover, f"HTML 里残留未替换的模板占位符: {leftover}"

    def test_render_with_errors_shows_error_summary(self, tmp_path):
        """有错误时 HTML/MD 里含错误信息。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["errors"] = ["renderer: RuntimeError: 渲染失败"]
        html_path, md_path = render(results, tmp_path)
        html_content = html_path.read_text(encoding="utf-8")
        md_content = md_path.read_text(encoding="utf-8")
        assert "RuntimeError" in html_content or "RuntimeError" in md_content


# ---------------------------------------------------------------------------
# _build_value_picks wiring unit tests
# ---------------------------------------------------------------------------

class TestBuildValuePicks:

    def _make_scores_df(self) -> "pd.DataFrame":
        import pandas as pd
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2026-05-16"), "600519"),
                (pd.Timestamp("2026-05-16"), "000858"),
                (pd.Timestamp("2026-05-16"), "601012"),
            ],
            names=["date", "ticker"],
        )
        return pd.DataFrame(
            {
                "composite_score": [0.9, 0.7, 0.5],
                "value_score":     [0.8, 0.6, 0.4],
                "quality_score":   [0.75, 0.65, 0.3],
                "growth_score":    [0.5, 0.4, 0.6],
            },
            index=idx,
        )

    def test_returns_list_of_dicts(self):
        from astock_quant.predict.daily import _build_value_picks
        picks = _build_value_picks(self._make_scores_df(), None, "2026-05-16")
        assert isinstance(picks, list)
        assert len(picks) > 0
        assert all(isinstance(p, dict) for p in picks)

    def test_sorted_by_composite_score_desc(self):
        from astock_quant.predict.daily import _build_value_picks
        picks = _build_value_picks(self._make_scores_df(), None, "2026-05-16")
        scores = [p["composite_score"] for p in picks]
        assert scores == sorted(scores, reverse=True)

    def test_each_pick_has_required_keys(self):
        from astock_quant.predict.daily import _build_value_picks
        picks = _build_value_picks(self._make_scores_df(), None, "2026-05-16")
        for p in picks:
            for key in ("ticker", "composite_score", "reason"):
                assert key in p, f"pick missing key '{key}': {p}"

    def test_reason_string_nonempty(self):
        from astock_quant.predict.daily import _build_value_picks
        picks = _build_value_picks(self._make_scores_df(), None, "2026-05-16")
        for p in picks:
            assert isinstance(p["reason"], str) and p["reason"], \
                f"reason should be non-empty string, got: {p['reason']!r}"

    def test_empty_dataframe_returns_empty_list(self):
        import pandas as pd
        from astock_quant.predict.daily import _build_value_picks
        empty = pd.DataFrame(
            columns=["composite_score", "value_score", "quality_score", "growth_score"]
        )
        picks = _build_value_picks(empty, None, "2026-05-16")
        assert picks == []

    def test_top_n_respected(self):
        import pandas as pd
        from astock_quant.predict.daily import _build_value_picks
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-05-16"), f"6000{i:02d}") for i in range(30)],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(
            {
                "composite_score": [i / 30 for i in range(30)],
                "value_score":     [0.5] * 30,
                "quality_score":   [0.5] * 30,
                "growth_score":    [0.5] * 30,
            },
            index=idx,
        )
        picks = _build_value_picks(df, None, "2026-05-16", top_n=5)
        assert len(picks) <= 5

    def test_future_date_slices_latest_available(self):
        from astock_quant.predict.daily import _build_value_picks
        # Asking for a date after the data — should still return picks from last available
        picks = _build_value_picks(self._make_scores_df(), None, "2026-12-31")
        assert len(picks) > 0

    def test_none_scores_df_returns_empty(self):
        from astock_quant.predict.daily import _build_value_picks
        picks = _build_value_picks(None, None, "2026-05-16")
        assert picks == []

    def test_roe_shown_as_ttm_full_year_when_financials_given(self):
        """T10：传入 financials 时，§1 ROE 显示「全年口径」TTM ROE，不是单季数.

        构造一只票：最新报告期是 2026Q1，单季累计 ROE=3.5（季报口径，会误导用户）。
        提供完整 4 季 + 上年同期记录让 TTM 算得出：
          TTM ROE(2026Q1) = 3.5 + 2025全年(15.0) - 2025Q1(3.6) = 14.9（全年量级）。
        断言 pick 里的 roe ≈ 14.9，不是 3.5。
        """
        import pandas as pd
        from astock_quant.contracts import FinancialMetrics
        from astock_quant.predict.daily import _build_value_picks

        scores = pd.DataFrame(
            {
                "composite_score": [0.9],
                "value_score": [0.8],
                "quality_score": [0.75],
                "growth_score": [0.5],
            },
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-05-16"), "601838")], names=["date", "ticker"]
            ),
        )
        # 季报口径 ROE 累计 YTD：单季 3.5、上年同期 3.6、上年全年 15.0
        recs = [
            FinancialMetrics(ticker="601838", report_period="20250331",
                             publish_date="20250430", roe=3.6),
            FinancialMetrics(ticker="601838", report_period="20251231",
                             publish_date="20260430", roe=15.0),
            FinancialMetrics(ticker="601838", report_period="20260331",
                             publish_date="20260430", roe=3.5),
        ]
        picks = _build_value_picks(
            scores, None, "2026-05-16", financials={"601838": recs}
        )
        assert len(picks) == 1
        roe = picks[0]["roe"]
        assert roe is not None
        # TTM = 3.5 + 15.0 - 3.6 = 14.9（全年量级），不是单季 3.5
        assert abs(roe - 14.9) < 1e-6, f"§1 ROE 应是全年口径 TTM 14.9，实际 {roe}"

    def test_roe_ttm_respects_publish_date(self):
        """T10：TTM ROE 展示严守披露日 —— 年报披露前不用它.

        站在 2026-03-15：2025 年报（publish_date=2026-04-30）还没披露，最新可见的是
        2025Q3。所以这天的 TTM ROE 应基于 2025Q3，而不是用上 2025 年报数据。
        """
        import pandas as pd
        from astock_quant.contracts import FinancialMetrics
        from astock_quant.predict.daily import _build_value_picks

        scores = pd.DataFrame(
            {
                "composite_score": [0.9], "value_score": [0.8],
                "quality_score": [0.75], "growth_score": [0.5],
            },
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-03-15"), "601838")], names=["date", "ticker"]
            ),
        )
        # 2025 三季报（披露 2025-10-31，3-15 前已可见）+ 2025 年报（披露 2026-04-30，未来）
        recs = [
            FinancialMetrics(ticker="601838", report_period="20240930",
                             publish_date="20241031", roe=13.0),
            FinancialMetrics(ticker="601838", report_period="20241231",
                             publish_date="20250430", roe=17.0),
            FinancialMetrics(ticker="601838", report_period="20250930",
                             publish_date="20251031", roe=12.0),
            FinancialMetrics(ticker="601838", report_period="20251231",
                             publish_date="20260430", roe=15.0),  # 未来，不应被用
        ]
        picks = _build_value_picks(
            scores, None, "2026-03-15", financials={"601838": recs}
        )
        assert len(picks) == 1
        # 站在 3-15，最新可见是 2025Q3 → TTM = 12.0 + 17.0 - 13.0 = 16.0
        # 若误用 2025 年报会得 15.0（look-ahead）
        roe = picks[0]["roe"]
        assert roe is not None
        assert abs(roe - 16.0) < 1e-6, (
            f"3-15 时 TTM ROE 应基于 2025Q3（=16.0），实际 {roe} —— "
            "若为 15.0 说明误用了未披露的 2025 年报"
        )


# ---------------------------------------------------------------------------
# 价值选股推荐名单渲染
# ---------------------------------------------------------------------------

def _make_value_picks() -> list[dict]:
    return [
        {
            "ticker": "600519",
            "composite_score": 0.85,
            "pe_percentile": 12.0,
            "pb_percentile": 8.5,
            "roe": 28.3,
            "reason": "PE 历史低位，ROE 行业第一",
        },
        {
            "ticker": "000858",
            "composite_score": 0.72,
            "pe_percentile": 25.0,
            "pb_percentile": 18.0,
            "roe": 22.1,
            "reason": "估值合理，持续盈利能力强",
        },
    ]


def _make_backtest() -> dict:
    return {
        "strategy_total_return": 0.423,
        "benchmark_total_return": 0.187,
        "excess_return": 0.236,
        "sharpe_ratio": 1.24,
        "max_drawdown": -0.158,
        "n_quarters": 8,
        "period": "2022-01-01 ~ 2024-01-01",
        "caveat": "回测不代表实盘，历史收益不预测未来。",
    }


class TestValuePicksRenderer:

    def test_value_picks_html_renders_ticker(self, tmp_path):
        """有 value_picks 时 HTML 包含股票代码。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["value_picks"] = _make_value_picks()
        html_path, _ = render(results, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        assert "600519" in content

    def test_value_picks_md_renders_ticker(self, tmp_path):
        """有 value_picks 时 MD 包含股票代码。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["value_picks"] = _make_value_picks()
        _, md_path = render(results, tmp_path)
        content = md_path.read_text(encoding="utf-8")
        assert "600519" in content

    def test_value_picks_html_shows_score(self, tmp_path):
        """HTML 里显示综合分。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["value_picks"] = _make_value_picks()
        html_path, _ = render(results, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        assert "0.850" in content

    def test_value_picks_none_shows_placeholder(self, tmp_path):
        """value_picks=None 时报告显示占位提示，不崩溃。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["value_picks"] = None
        html_path, md_path = render(results, tmp_path)
        html_content = html_path.read_text(encoding="utf-8")
        md_content = md_path.read_text(encoding="utf-8")
        assert "尚未就绪" in html_content or "尚未就绪" in md_content

    def test_value_picks_missing_key_no_crash(self, tmp_path):
        """value_picks 条目缺字段时不崩溃（容错）。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["value_picks"] = [{"ticker": "600519"}]
        html_path, md_path = render(results, tmp_path)
        assert html_path.exists()
        assert md_path.exists()

    def test_value_picks_html_contains_reason(self, tmp_path):
        """入选理由出现在 HTML 报告里。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["value_picks"] = _make_value_picks()
        html_path, _ = render(results, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        assert "ROE 行业第一" in content


class TestBacktestRenderer:

    def test_backtest_html_renders_strategy_return(self, tmp_path):
        """有 backtest 时 HTML 包含策略收益数字。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["backtest"] = _make_backtest()
        html_path, _ = render(results, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        assert "42.3%" in content

    def test_backtest_md_renders_benchmark(self, tmp_path):
        """有 backtest 时 MD 包含沪深300 基准收益。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["backtest"] = _make_backtest()
        _, md_path = render(results, tmp_path)
        content = md_path.read_text(encoding="utf-8")
        assert "18.7%" in content

    def test_backtest_none_shows_placeholder(self, tmp_path):
        """backtest=None 时报告显示占位提示，不崩溃。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["backtest"] = None
        html_path, md_path = render(results, tmp_path)
        html_content = html_path.read_text(encoding="utf-8")
        md_content = md_path.read_text(encoding="utf-8")
        assert "尚未就绪" in html_content or "尚未就绪" in md_content

    def test_backtest_html_contains_caveat(self, tmp_path):
        """回测免责声明出现在 HTML 里。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["backtest"] = _make_backtest()
        html_path, _ = render(results, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        assert "回测不代表实盘" in content

    def test_backtest_excess_return_in_md(self, tmp_path):
        """超额收益出现在 MD 里。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["backtest"] = _make_backtest()
        _, md_path = render(results, tmp_path)
        content = md_path.read_text(encoding="utf-8")
        assert "23.6%" in content


class TestNewReportStructure:
    """价值选股报告结构 —— 2026-05-22 用户决策后旧涨跌预测章节已整段移除。

    报告结构（HTML/MD 一致）：今日速览 → 诚信声明 → §1 价值名单 → §2 回测
    → §3 历史准确率 → §4 元数据。
    """

    def test_html_value_section_is_section_1(self, tmp_path):
        """HTML 里价值选股是 §1，排在回测 §2 之前。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["value_picks"] = _make_value_picks()
        html_path, _ = render(results, tmp_path)
        content = html_path.read_text(encoding="utf-8")
        idx_value = content.find("§1 本季度价值选股推荐名单")
        idx_backtest = content.find("§2 策略回测")
        assert idx_value != -1, "HTML 缺少 §1 价值选股名单"
        assert idx_backtest != -1, "HTML 缺少 §2 回测"
        assert idx_value < idx_backtest, "价值选股 §1 应在回测 §2 之前"

    def test_md_value_section_is_section_1(self, tmp_path):
        """MD 里价值选股是 §1，排在回测 §2 之前。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["value_picks"] = _make_value_picks()
        _, md_path = render(results, tmp_path)
        content = md_path.read_text(encoding="utf-8")
        idx_value = content.find("§1 本季度价值选股推荐名单")
        idx_backtest = content.find("§2 策略回测")
        assert idx_value != -1, "MD 缺少 §1 价值选股名单"
        assert idx_backtest != -1, "MD 缺少 §2 回测"
        assert idx_value < idx_backtest, "价值选股 §1 应在回测 §2 之前"

    def test_old_prediction_sections_removed(self, tmp_path):
        """命门：旧涨跌预测痕迹彻底清除 —— 不再有「实验性预测」「明日强势评分」等章节。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["value_picks"] = _make_value_picks()
        html_path, md_path = render(results, tmp_path)
        for content, name in [
            (html_path.read_text(encoding="utf-8"), "HTML"),
            (md_path.read_text(encoding="utf-8"), "MD"),
        ]:
            assert "短期实验性预测" not in content, f"{name} 仍残留「短期实验性预测」章节"
            assert "接近随机基线" not in content, f"{name} 仍残留旧预测降级说明"
            assert "DirectionModel" not in content, f"{name} 仍残留旧模型名 DirectionModel"

    def test_sections_numbered_consecutively(self, tmp_path):
        """章节编号 §1-§4 连续无跳号（HTML 与 MD 一致）。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        results["value_picks"] = _make_value_picks()
        html_path, md_path = render(results, tmp_path)
        for content, name in [
            (html_path.read_text(encoding="utf-8"), "HTML"),
            (md_path.read_text(encoding="utf-8"), "MD"),
        ]:
            for sec in ("§1", "§2", "§3", "§4"):
                assert sec in content, f"{name} 缺少章节 {sec}"
            assert "§5" not in content, f"{name} 不应再有 §5（旧预测章节已删）"

    def test_disclaimer_still_present_with_new_structure(self, tmp_path):
        """命门：诚信声明在新报告结构里仍存在（不构成投资建议）。"""
        from astock_quant.predict.renderer import render
        results = _make_full_results()
        html_path, md_path = render(results, tmp_path)
        html_content = html_path.read_text(encoding="utf-8")
        md_content = md_path.read_text(encoding="utf-8")
        for content, name in [(html_content, "HTML"), (md_content, "MD")]:
            has_disclaimer = (
                "诚信声明" in content
                or "不构成投资" in content
            )
            assert has_disclaimer, f"命门失败：{name} 新结构里缺少诚信声明"


# ---------------------------------------------------------------------------
# 集成测试：_try_build_value_picks 用 compute_factor_frame 产出非空 picks
# ---------------------------------------------------------------------------

class TestTryBuildValuePicksIntegration:
    """Guard that _try_build_value_picks produces real picks via compute_factor_frame.

    Uses real factor/value_score modules but injects minimal synthetic data so the
    test does not require network access or on-disk artifacts.
    """

    def _make_price_panel(self) -> pd.DataFrame:
        """Build a MultiIndex=(date, ticker) price panel matching compute_factor_frame's expectation."""
        import numpy as np
        dates = pd.date_range("2026-01-01", periods=60, freq="B")
        tickers = [f"60{i:04d}" for i in range(10)]
        rng = np.random.default_rng(42)
        frames = []
        for t in tickers:
            base = 10 + rng.random() * 90
            prices = base + rng.random(len(dates)).cumsum() * 0.1
            df = pd.DataFrame({
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "volume": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
                "pe": 8 + rng.random(len(dates)) * 20,
                "pb": 0.5 + rng.random(len(dates)) * 3,
                "dividend_yield": rng.random(len(dates)) * 0.05,
            }, index=dates)
            df.index.name = "date"
            df["ticker"] = t
            frames.append(df)
        panel = pd.concat(frames).reset_index().set_index(["date", "ticker"]).sort_index()
        return panel

    def _make_financials(self, tickers: list) -> dict:
        import numpy as np
        from astock_quant.contracts import FinancialMetrics
        rng = np.random.default_rng(42)
        fin = {}
        fin_dates = pd.date_range("2025-01-01", periods=4, freq="QE")
        for t in tickers:
            records = []
            for d in fin_dates:
                records.append(FinancialMetrics(
                    ticker=t,
                    report_period=d.strftime("%Y%m%d"),
                    publish_date=d.strftime("%Y%m%d"),
                    roe=0.05 + rng.random() * 0.25,
                    gross_margin=0.1 + rng.random() * 0.5,
                    net_margin=0.05 + rng.random() * 0.3,
                ))
            fin[t] = records
        return fin

    def test_try_build_value_picks_returns_nonempty_list(self):
        """§1 核心命门：_try_build_value_picks 必须返回非空 list，不得返回 None。"""
        from astock_quant.predict.daily import _try_build_value_picks

        price_panel = self._make_price_panel()
        tickers = price_panel.index.get_level_values("ticker").unique().tolist()
        financials = self._make_financials(tickers)
        prepared_data = {"prices": price_panel, "financials": financials, "moneyflow": None}

        picks = _try_build_value_picks(
            universe=tickers,
            date_str="2026-02-28",
            prepared_data=prepared_data,
        )

        assert picks is not None, "§1 命门：_try_build_value_picks 返回 None，报告会显示占位"
        assert len(picks) > 0, "§1 命门：value_picks 列表为空，报告会显示空名单"
        assert all("ticker" in p and "composite_score" in p for p in picks)

    def test_try_build_value_picks_sorted_desc(self):
        """picks 按 composite_score 降序排列。"""
        from astock_quant.predict.daily import _try_build_value_picks

        price_panel = self._make_price_panel()
        tickers = price_panel.index.get_level_values("ticker").unique().tolist()
        financials = self._make_financials(tickers)
        prepared_data = {"prices": price_panel, "financials": financials, "moneyflow": None}

        picks = _try_build_value_picks(
            universe=tickers, date_str="2026-02-28", prepared_data=prepared_data
        )
        if picks and len(picks) > 1:
            scores = [p["composite_score"] for p in picks]
            assert scores == sorted(scores, reverse=True)
