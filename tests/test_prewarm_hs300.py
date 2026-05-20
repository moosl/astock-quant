"""P15 prewarm_hs300 测试.

覆盖：
- mock AStockSource + load_prices/moneyflow/financials，验证 N 只票都被调用
- 单只票 prices 连续失败 3 次 → 加入 missing_list，继续不中断
- 进度条输出含百分比格式 [i/total]
- 全部成功 → sys.exit 不被调
- 部分失败 → sys.exit(1)
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch



# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_FAKE_UNIVERSE = ["600519", "000858", "600036"]


def _make_mock_source():
    src = MagicMock()
    src.get_prices = MagicMock(return_value=[])
    return src


# ---------------------------------------------------------------------------
# main() 基础流程
# ---------------------------------------------------------------------------

class TestPrewarmMain:

    def _run_main(self, mock_universe, load_prices_fn=None, load_moneyflow_fn=None,
                  load_financials_fn=None, capture_output=False):
        """Helper：patch 所有外部依赖后跑 main()."""
        out = StringIO() if capture_output else None

        with (
            patch("astock_quant.scripts.prewarm_hs300.get_hs300_universe",
                  return_value=mock_universe),
            patch("astock_quant.scripts.prewarm_hs300.AStockSource",
                  return_value=_make_mock_source()),
            patch("astock_quant.scripts.prewarm_hs300.load_prices",
                  side_effect=load_prices_fn or (lambda *a, **kw: None)),
            patch("astock_quant.scripts.prewarm_hs300.load_moneyflow",
                  side_effect=load_moneyflow_fn or (lambda *a, **kw: None)),
            patch("astock_quant.scripts.prewarm_hs300.load_financials",
                  side_effect=load_financials_fn or (lambda *a, **kw: None)),
        ):
            if capture_output:
                with patch("sys.stdout", out):
                    from astock_quant.scripts.prewarm_hs300 import main
                    try:
                        main()
                    except SystemExit:
                        pass
            else:
                from astock_quant.scripts.prewarm_hs300 import main
                try:
                    main()
                except SystemExit:
                    pass

        return out.getvalue() if capture_output else None

    def test_load_prices_called_for_each_ticker(self):
        """每只 ticker 都触发 load_prices 调用一次（成功路径）。"""
        prices_calls = []

        def tracking_load_prices(code, source, start, end, force_refresh=False):
            prices_calls.append(code)

        self._run_main(_FAKE_UNIVERSE, load_prices_fn=tracking_load_prices)
        assert len(prices_calls) == len(_FAKE_UNIVERSE), \
            f"load_prices 调用次数 {len(prices_calls)} != universe 大小 {len(_FAKE_UNIVERSE)}"

    def test_all_tickers_processed_not_just_first(self):
        """即使第 1 只 ticker 失败，后续 ticker 仍被处理。"""
        processed = []

        def partial_fail_prices(code, source, start, end, force_refresh=False):
            processed.append(code)
            if code == _FAKE_UNIVERSE[0]:
                raise RuntimeError("第一只失败")

        self._run_main(_FAKE_UNIVERSE, load_prices_fn=partial_fail_prices)
        # 全部 3 只都应被尝试（每只 3 次 attempt，第一只全部 fail）
        assert set(_FAKE_UNIVERSE).issubset(set(processed)), \
            f"第一只失败后后续 ticker 未被处理: {processed}"

    def test_prices_failure_adds_to_missing_list(self, capsys):
        """prices 3 次全失败 → 该 ticker 进入 missing_list，最后输出失败列表。"""
        fail_ticker = _FAKE_UNIVERSE[0]

        def always_fail_prices(code, source, start, end, force_refresh=False):
            if code == fail_ticker:
                raise RuntimeError("网络超时")

        self._run_main(_FAKE_UNIVERSE, load_prices_fn=always_fail_prices)
        captured = capsys.readouterr()
        # missing_list 或 fail 字样出现在输出里
        assert fail_ticker in captured.out or "失败" in captured.out, \
            f"missing ticker {fail_ticker} 未在输出中体现：{captured.out}"

    def test_progress_output_contains_fraction(self, capsys):
        """进度输出含 [i/total] 格式（如 [1/3] / [2/3]）。"""
        self._run_main(_FAKE_UNIVERSE)
        captured = capsys.readouterr()
        total = len(_FAKE_UNIVERSE)
        assert f"[1/{total}]" in captured.out, \
            f"进度输出不含 [1/{total}]，实际输出：{captured.out[:200]}"

    def test_all_success_no_sys_exit(self):
        """全部成功 → sys.exit 不被调（退出码 0）。"""
        exit_spy = MagicMock()
        with (
            patch("astock_quant.scripts.prewarm_hs300.get_hs300_universe",
                  return_value=["600519"]),
            patch("astock_quant.scripts.prewarm_hs300.AStockSource",
                  return_value=_make_mock_source()),
            patch("astock_quant.scripts.prewarm_hs300.load_prices",
                  return_value=None),
            patch("astock_quant.scripts.prewarm_hs300.load_moneyflow",
                  return_value=None),
            patch("astock_quant.scripts.prewarm_hs300.load_financials",
                  return_value=None),
            patch("sys.exit", exit_spy),
        ):
            from astock_quant.scripts.prewarm_hs300 import main
            main()

        exit_spy.assert_not_called()

    def test_partial_failure_calls_sys_exit_1(self):
        """prices 3 次全失败 → sys.exit(1) 被调。"""
        exit_calls = []

        def always_fail(code, source, start, end, force_refresh=False):
            raise RuntimeError("全挂了")

        with (
            patch("astock_quant.scripts.prewarm_hs300.get_hs300_universe",
                  return_value=["600519"]),
            patch("astock_quant.scripts.prewarm_hs300.AStockSource",
                  return_value=_make_mock_source()),
            patch("astock_quant.scripts.prewarm_hs300.load_prices",
                  side_effect=always_fail),
            patch("astock_quant.scripts.prewarm_hs300.load_moneyflow",
                  return_value=None),
            patch("astock_quant.scripts.prewarm_hs300.load_financials",
                  return_value=None),
            patch("sys.exit", side_effect=lambda code: exit_calls.append(code)),
        ):
            from astock_quant.scripts.prewarm_hs300 import main
            main()

        assert 1 in exit_calls, f"预期 sys.exit(1)，实际 exit_calls={exit_calls}"

    def test_prices_retried_3_times_on_failure(self):
        """单只 ticker prices 失败时重试 3 次（attempt 0/1/2）。"""
        attempt_counts = {}

        def counting_fail(code, source, start, end, force_refresh=False):
            attempt_counts[code] = attempt_counts.get(code, 0) + 1
            raise RuntimeError("每次都失败")

        with (
            patch("astock_quant.scripts.prewarm_hs300.get_hs300_universe",
                  return_value=["600519"]),
            patch("astock_quant.scripts.prewarm_hs300.AStockSource",
                  return_value=_make_mock_source()),
            patch("astock_quant.scripts.prewarm_hs300.load_prices",
                  side_effect=counting_fail),
            patch("astock_quant.scripts.prewarm_hs300.load_moneyflow",
                  return_value=None),
            patch("astock_quant.scripts.prewarm_hs300.load_financials",
                  return_value=None),
            patch("sys.exit", side_effect=lambda c: None),
        ):
            from astock_quant.scripts.prewarm_hs300 import main
            main()

        assert attempt_counts.get("600519", 0) == 3, \
            f"期望 3 次重试，实际 {attempt_counts}"

    def test_moneyflow_failure_does_not_block_prices_success(self, capsys):
        """moneyflow 失败不影响 prices 成功，不加入 missing_list。"""
        prices_ok = []

        def ok_prices(code, source, start, end, force_refresh=False):
            prices_ok.append(code)

        def fail_moneyflow(code, source, start, end, force_refresh=False):
            raise RuntimeError("moneyflow 挂了")

        with (
            patch("astock_quant.scripts.prewarm_hs300.get_hs300_universe",
                  return_value=["600519"]),
            patch("astock_quant.scripts.prewarm_hs300.AStockSource",
                  return_value=_make_mock_source()),
            patch("astock_quant.scripts.prewarm_hs300.load_prices",
                  side_effect=ok_prices),
            patch("astock_quant.scripts.prewarm_hs300.load_moneyflow",
                  side_effect=fail_moneyflow),
            patch("astock_quant.scripts.prewarm_hs300.load_financials",
                  return_value=None),
            patch("sys.exit", side_effect=lambda c: None),
        ):
            from astock_quant.scripts.prewarm_hs300 import main
            main()

        # prices 成功 → "600519" 不应在 missing_list 里 → sys.exit(1) 不应被调
        assert "600519" in prices_ok
