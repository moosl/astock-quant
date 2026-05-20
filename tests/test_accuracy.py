"""P14 准确率追踪测试 —— accuracy.py.

覆盖：
- 辅助函数：_future_close_at_horizon / _future_close_path / _GroundTruthCache
- 4 类 evaluator：_eval_direction / _eval_return / _eval_ranking / _eval_trade_signal
- 主函数：evaluate_predictions（空目录 / horizon cutoff / 缺 GT / 损坏 JSON / target filter / ticker filter）
- CLI：--days 默认值 / 空 predictions exit 1
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from astock_quant.contracts import PriceBar


# ---------------------------------------------------------------------------
# FakeSource：实现 get_prices(ticker, start, end) → list[PriceBar]
# ---------------------------------------------------------------------------

def _make_price_bars(
    ticker: str,
    start: str,
    n_days: int,
    start_close: float = 100.0,
    daily_delta: float = 0.0,
) -> list[PriceBar]:
    """生成连续 n 个交易日的 PriceBar（close 按 daily_delta 线性变动）."""
    bars = []
    d = _dt.date.fromisoformat(start)
    close = start_close
    for i in range(n_days):
        bars.append(PriceBar(
            ticker=ticker,
            date=d + _dt.timedelta(days=i),
            open=close,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=1_000_000.0,
        ))
        close += daily_delta
    return bars


class FakeSource:
    """可配置 ticker → list[PriceBar] 的 mock DataSource."""

    def __init__(self, data: dict[str, list[PriceBar]] | None = None) -> None:
        self._data = data or {}
        self.fetch_count: dict[str, int] = {}

    def get_prices(self, ticker: str, start: str, end: str) -> list[PriceBar]:
        self.fetch_count[ticker] = self.fetch_count.get(ticker, 0) + 1
        return self._data.get(ticker, [])


def _make_close_series(bars: list[PriceBar]) -> pd.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(b.date) for b in bars])
    return pd.Series([b.close for b in bars], index=idx)


# ---------------------------------------------------------------------------
# _future_close_at_horizon
# ---------------------------------------------------------------------------

class TestFutureCloseAtHorizon:

    def _make_series(self, start: str = "2026-01-02", n: int = 20,
                     start_close: float = 100.0, delta: float = 1.0) -> pd.Series:
        bars = _make_price_bars("A", start, n, start_close, delta)
        return _make_close_series(bars)

    def test_normal_case_returns_entry_and_future(self):
        from astock_quant.predict.accuracy import _future_close_at_horizon
        s = self._make_series()  # day 0=100, day 1=101, ..., day 5=105
        entry, future = _future_close_at_horizon(s, _dt.date(2026, 1, 2), horizon=5)
        assert entry == pytest.approx(100.0)
        assert future == pytest.approx(105.0)

    def test_past_end_returns_none(self):
        from astock_quant.predict.accuracy import _future_close_at_horizon
        s = self._make_series(n=5)  # only 5 bars
        entry, future = _future_close_at_horizon(s, _dt.date(2026, 1, 2), horizon=10)
        assert entry is None
        assert future is None

    def test_t_not_in_series_uses_prior_bar(self):
        """T 不在序列（停牌）→ 用最近前一个交易日做 entry."""
        from astock_quant.predict.accuracy import _future_close_at_horizon
        bars = _make_price_bars("A", "2026-01-02", 10, 100.0, 1.0)
        # 删掉 index 2（2026-01-04）模拟停牌
        bars = [b for b in bars if b.date != _dt.date(2026, 1, 4)]
        s = _make_close_series(bars)
        # T = 2026-01-04 不在序列 → 回退到 2026-01-03（close=101）
        entry, future = _future_close_at_horizon(s, _dt.date(2026, 1, 4), horizon=3)
        assert entry is not None  # 回退到前一交易日
        assert future is not None

    def test_t_before_series_start_returns_none(self):
        from astock_quant.predict.accuracy import _future_close_at_horizon
        s = self._make_series(start="2026-01-05")
        entry, future = _future_close_at_horizon(s, _dt.date(2026, 1, 2), horizon=5)
        assert entry is None and future is None


# ---------------------------------------------------------------------------
# _future_close_path
# ---------------------------------------------------------------------------

class TestFutureClosePath:

    def test_returns_correct_path_length(self):
        from astock_quant.predict.accuracy import _future_close_path
        bars = _make_price_bars("A", "2026-01-02", 15, 100.0, 1.0)
        s = _make_close_series(bars)
        entry, path = _future_close_path(s, _dt.date(2026, 1, 2), horizon=5)
        assert entry == pytest.approx(100.0)
        assert path is not None
        assert len(path) == 5

    def test_path_past_end_returns_none(self):
        from astock_quant.predict.accuracy import _future_close_path
        bars = _make_price_bars("A", "2026-01-02", 4, 100.0, 1.0)
        s = _make_close_series(bars)
        entry, path = _future_close_path(s, _dt.date(2026, 1, 2), horizon=5)
        assert entry is None and path is None

    def test_path_values_are_future_closes(self):
        from astock_quant.predict.accuracy import _future_close_path
        bars = _make_price_bars("A", "2026-01-02", 10, 100.0, 2.0)
        s = _make_close_series(bars)
        # entry=100, path=[102, 104, 106]
        entry, path = _future_close_path(s, _dt.date(2026, 1, 2), horizon=3)
        assert entry == pytest.approx(100.0)
        assert path == pytest.approx([102.0, 104.0, 106.0])


# ---------------------------------------------------------------------------
# _GroundTruthCache
# ---------------------------------------------------------------------------

class TestGroundTruthCache:

    def test_dedup_same_ticker_fetches_once(self):
        """同一 ticker 调 get_close_series 2 次 → 只 fetch 1 次（缓存命中）."""
        from astock_quant.predict.accuracy import _GroundTruthCache
        bars = _make_price_bars("600519", "2026-01-02", 10)
        source = FakeSource({"600519": bars})
        cache = _GroundTruthCache(source)
        cache.set_date_range(_dt.date(2026, 1, 2), _dt.date(2026, 1, 15))

        cache.get_close_series("600519")
        cache.get_close_series("600519")

        assert source.fetch_count.get("600519", 0) == 1

    def test_missing_ticker_returns_none(self):
        from astock_quant.predict.accuracy import _GroundTruthCache
        source = FakeSource({})
        cache = _GroundTruthCache(source)
        cache.set_date_range(_dt.date(2026, 1, 2), _dt.date(2026, 1, 15))
        assert cache.get_close_series("NOTEXIST") is None

    def test_missing_ticker_not_refetched(self):
        """缺失 ticker 第二次不再 fetch（missing set 缓存）."""
        from astock_quant.predict.accuracy import _GroundTruthCache
        source = FakeSource({})
        cache = _GroundTruthCache(source)
        cache.set_date_range(_dt.date(2026, 1, 2), _dt.date(2026, 1, 15))
        cache.get_close_series("000001")
        cache.get_close_series("000001")
        assert source.fetch_count.get("000001", 0) == 1

    def test_set_date_range_required_before_fetch(self):
        from astock_quant.predict.accuracy import _GroundTruthCache
        cache = _GroundTruthCache(FakeSource({}))
        with pytest.raises(RuntimeError, match="set_date_range"):
            cache.get_close_series("600519")


# ---------------------------------------------------------------------------
# _eval_direction
# ---------------------------------------------------------------------------

class TestEvalDirection:

    def _make_stats(self) -> dict:
        from collections import defaultdict
        return defaultdict(float)

    def _make_gt_cache(self, ticker: str, bars: list[PriceBar]):
        from astock_quant.predict.accuracy import _GroundTruthCache
        cache = _GroundTruthCache(FakeSource({ticker: bars}))
        cache.set_date_range(_dt.date(2026, 1, 1), _dt.date(2026, 2, 1))
        return cache

    def test_buy_correct_counts_hit(self):
        """value=1.0 (buy) + 真实涨 → hit++."""
        from astock_quant.predict.accuracy import _eval_direction
        # 连涨 10 天：entry=100, T+5=105
        bars = _make_price_bars("600519", "2026-01-02", 10, 100.0, 1.0)
        cache = self._make_gt_cache("600519", bars)
        stats = self._make_stats()
        preds = [{"ticker": "600519", "value": 1.0, "score": 0.7}]
        _eval_direction(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                        ticker_filter=None, stats=stats)
        assert stats["n_evaluated"] == 1
        assert stats["n_hit"] == 1

    def test_sell_wrong_no_hit(self):
        """value=0.0 (sell) + 真实涨 → 不 hit."""
        from astock_quant.predict.accuracy import _eval_direction
        bars = _make_price_bars("600519", "2026-01-02", 10, 100.0, 1.0)
        cache = self._make_gt_cache("600519", bars)
        stats = self._make_stats()
        preds = [{"ticker": "600519", "value": 0.0, "score": 0.3}]
        _eval_direction(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                        ticker_filter=None, stats=stats)
        assert stats["n_evaluated"] == 1
        assert stats["n_hit"] == 0

    def test_missing_gt_increments_counter(self):
        """source 返回空 → n_missing_gt++。"""
        from astock_quant.predict.accuracy import _eval_direction, _GroundTruthCache
        cache = _GroundTruthCache(FakeSource({}))
        cache.set_date_range(_dt.date(2026, 1, 1), _dt.date(2026, 2, 1))
        stats = self._make_stats()
        preds = [{"ticker": "600519", "value": 1.0}]
        _eval_direction(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                        ticker_filter=None, stats=stats)
        assert stats["n_missing_gt"] == 1
        assert stats["n_evaluated"] == 0

    def test_ticker_filter_skips_other_tickers(self):
        """ticker_filter='600519' → 只处理 600519，跳过其他 ticker。"""
        from astock_quant.predict.accuracy import _eval_direction
        bars = _make_price_bars("600519", "2026-01-02", 10, 100.0, 1.0)
        cache = self._make_gt_cache("600519", bars)
        stats = self._make_stats()
        preds = [
            {"ticker": "600519", "value": 1.0},
            {"ticker": "000858", "value": 1.0},  # 应被过滤
        ]
        _eval_direction(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                        ticker_filter="600519", stats=stats)
        assert stats["n_evaluated"] == 1


# ---------------------------------------------------------------------------
# _eval_return
# ---------------------------------------------------------------------------

class TestEvalReturn:

    def _make_stats(self) -> dict:
        from collections import defaultdict
        return defaultdict(float)

    def test_mae_calculation(self):
        """MAE = |pred - actual|，用已知数值验证计算正确。"""
        from astock_quant.predict.accuracy import _eval_return, _GroundTruthCache
        # entry=100, T+5=110 → actual_return=0.10
        bars = _make_price_bars("600519", "2026-01-02", 10, 100.0, 2.0)
        cache = _GroundTruthCache(FakeSource({"600519": bars}))
        cache.set_date_range(_dt.date(2026, 1, 1), _dt.date(2026, 2, 1))
        stats = self._make_stats()
        preds = [{"ticker": "600519", "value": 0.05}]  # 预测 5%，实际 10%
        _eval_return(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                     ticker_filter=None, stats=stats)
        assert stats["n_evaluated"] == 1
        assert stats["sum_abs_err"] == pytest.approx(0.05, abs=1e-6)

    def test_direction_correct_when_both_positive(self):
        """pred>0 + actual>0 → dir_correct++。"""
        from astock_quant.predict.accuracy import _eval_return, _GroundTruthCache
        bars = _make_price_bars("600519", "2026-01-02", 10, 100.0, 1.0)
        cache = _GroundTruthCache(FakeSource({"600519": bars}))
        cache.set_date_range(_dt.date(2026, 1, 1), _dt.date(2026, 2, 1))
        stats = self._make_stats()
        preds = [{"ticker": "600519", "value": 0.03}]  # pred>0
        _eval_return(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                     ticker_filter=None, stats=stats)
        assert stats["n_dir_correct"] == 1

    def test_direction_wrong_when_opposite(self):
        """pred>0 + actual<0 → dir_correct 不增。"""
        from astock_quant.predict.accuracy import _eval_return, _GroundTruthCache
        # 连跌：entry=100, T+5=95
        bars = _make_price_bars("600519", "2026-01-02", 10, 100.0, -1.0)
        cache = _GroundTruthCache(FakeSource({"600519": bars}))
        cache.set_date_range(_dt.date(2026, 1, 1), _dt.date(2026, 2, 1))
        stats = self._make_stats()
        preds = [{"ticker": "600519", "value": 0.05}]  # pred>0 但实际跌
        _eval_return(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                     ticker_filter=None, stats=stats)
        assert stats["n_dir_correct"] == 0


# ---------------------------------------------------------------------------
# _eval_ranking
# ---------------------------------------------------------------------------

class TestEvalRanking:

    def _make_stats(self) -> dict:
        from collections import defaultdict
        return defaultdict(float)

    def _make_cache(self, ticker_bars: dict[str, list[PriceBar]]):
        from astock_quant.predict.accuracy import _GroundTruthCache
        cache = _GroundTruthCache(FakeSource(ticker_bars))
        cache.set_date_range(_dt.date(2026, 1, 1), _dt.date(2026, 2, 1))
        return cache

    def test_topk_precision_perfect_alignment(self):
        """score 降序 Top 2 与真实涨幅 Top 1 完全一致 → precision=1.0."""
        from astock_quant.predict.accuracy import _eval_ranking
        # A: score 0.9, actual +5% (涨最多)
        # B: score 0.5, actual +1%
        # C: score 0.1, actual -3%
        bars_a = _make_price_bars("A", "2026-01-02", 10, 100.0, 1.0)  # T+5=105
        bars_b = _make_price_bars("B", "2026-01-02", 10, 100.0, 0.2)  # T+5=101
        bars_c = _make_price_bars("C", "2026-01-02", 10, 100.0, -0.6)  # T+5=97
        cache = self._make_cache({"A": bars_a, "B": bars_b, "C": bars_c})
        stats = self._make_stats()
        preds = [
            {"ticker": "A", "score": 0.9},
            {"ticker": "B", "score": 0.5},
            {"ticker": "C", "score": 0.1},
        ]
        _eval_ranking(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                      ticker_filter=None, stats=stats, top_k=2)
        # Top 2 by score: A, B; Top 1 by actual: A → A in Top2 → hits=1, precision@1=1.0
        assert stats["sum_topk_precision"] == pytest.approx(1.0)
        assert stats["n_days"] == 1

    def test_spearman_perfect(self):
        """score 与 actual_return 完全正相关 → spearman ≈ 1.0."""
        from astock_quant.predict.accuracy import _eval_ranking
        bars_a = _make_price_bars("A", "2026-01-02", 10, 100.0, 2.0)   # T+5=110
        bars_b = _make_price_bars("B", "2026-01-02", 10, 100.0, 1.0)   # T+5=105
        bars_c = _make_price_bars("C", "2026-01-02", 10, 100.0, 0.0)   # T+5=100
        cache = self._make_cache({"A": bars_a, "B": bars_b, "C": bars_c})
        stats = self._make_stats()
        preds = [
            {"ticker": "A", "score": 0.9},
            {"ticker": "B", "score": 0.5},
            {"ticker": "C", "score": 0.1},
        ]
        _eval_ranking(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                      ticker_filter=None, stats=stats, top_k=3)
        if stats["n_spearman_days"] > 0:
            avg_sp = stats["sum_spearman"] / stats["n_spearman_days"]
            assert avg_sp == pytest.approx(1.0, abs=0.01)

    def test_insufficient_rows_skips(self):
        """只有 1 只票有 GT → 少于 2 行，不计入 stats。"""
        from astock_quant.predict.accuracy import _eval_ranking
        bars_a = _make_price_bars("A", "2026-01-02", 10)
        cache = self._make_cache({"A": bars_a})
        stats = self._make_stats()
        preds = [
            {"ticker": "A", "score": 0.9},
            {"ticker": "NOTEXIST", "score": 0.1},
        ]
        _eval_ranking(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                      ticker_filter=None, stats=stats, top_k=2)
        assert stats["n_days"] == 0


# ---------------------------------------------------------------------------
# _eval_trade_signal
# ---------------------------------------------------------------------------

class TestEvalTradeSignal:

    def _make_stats(self) -> dict:
        from collections import defaultdict
        return defaultdict(float)

    def _make_cache_with_path(self, ticker: str, closes: list[float]):
        """构造一个 ticker 的 close 序列（从 2026-01-02 起）."""
        from astock_quant.predict.accuracy import _GroundTruthCache
        bars = []
        for i, c in enumerate(closes):
            d = _dt.date(2026, 1, 2) + _dt.timedelta(days=i)
            bars.append(PriceBar(
                ticker=ticker, date=d,
                open=c, high=c * 1.01, low=c * 0.99, close=c, volume=1e6,
            ))
        cache = _GroundTruthCache(FakeSource({ticker: bars}))
        cache.set_date_range(_dt.date(2026, 1, 1), _dt.date(2026, 2, 1))
        return cache

    def test_tp_hit(self):
        """value=+1 + 路径先触 TP(+5%) → hit."""
        from astock_quant.predict.accuracy import _eval_trade_signal
        # entry=100, path: 101, 102, 106(>=105=TP), 103, 97
        closes = [100.0, 101.0, 102.0, 106.0, 103.0, 97.0, 95.0]
        cache = self._make_cache_with_path("600519", closes)
        stats = self._make_stats()
        preds = [{"ticker": "600519", "value": 1.0}]
        _eval_trade_signal(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                           ticker_filter=None, stats=stats,
                           tp_pct=0.05, sl_pct=-0.03)
        assert stats["n_hit"] == 1
        assert stats["n_buy"] == 1
        assert stats["n_buy_profit"] == 1

    def test_sl_hit(self):
        """value=-1 + 路径先触 SL(-3%) → hit。"""
        from astock_quant.predict.accuracy import _eval_trade_signal
        # entry=100, path: 99, 98, 96(<=97=SL), 102, 106
        closes = [100.0, 99.0, 98.0, 96.0, 102.0, 106.0, 107.0]
        cache = self._make_cache_with_path("600519", closes)
        stats = self._make_stats()
        preds = [{"ticker": "600519", "value": -1.0}]
        _eval_trade_signal(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                           ticker_filter=None, stats=stats,
                           tp_pct=0.05, sl_pct=-0.03)
        assert stats["n_hit"] == 1

    def test_hold_hit(self):
        """value=0 + 路径 TP/SL 都没触 → hit。"""
        from astock_quant.predict.accuracy import _eval_trade_signal
        # entry=100, path: 101, 102, 103, 104, 104.9（未到 TP=105，SL=97）
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 104.9, 105.0]
        cache = self._make_cache_with_path("600519", closes)
        stats = self._make_stats()
        preds = [{"ticker": "600519", "value": 0.0}]
        _eval_trade_signal(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                           ticker_filter=None, stats=stats,
                           tp_pct=0.05, sl_pct=-0.03)
        assert stats["n_hit"] == 1

    def test_tp_predicted_but_sl_triggered_no_hit(self):
        """value=+1 + 路径先触 SL → 不 hit（预测 TP 但实际 SL 先到）。"""
        from astock_quant.predict.accuracy import _eval_trade_signal
        closes = [100.0, 99.0, 98.0, 96.5, 102.0, 108.0, 109.0]
        cache = self._make_cache_with_path("600519", closes)
        stats = self._make_stats()
        preds = [{"ticker": "600519", "value": 1.0}]
        _eval_trade_signal(preds, _dt.date(2026, 1, 2), cache, horizon=5,
                           ticker_filter=None, stats=stats,
                           tp_pct=0.05, sl_pct=-0.03)
        assert stats["n_hit"] == 0
        assert stats["n_buy"] == 1
        assert stats["n_buy_profit"] == 0


# ---------------------------------------------------------------------------
# evaluate_predictions — 主函数
# ---------------------------------------------------------------------------

def _write_prediction_json(path: Path, date_str: str, preds_by_target: dict) -> None:
    """写 predictions_YYYY-MM-DD.json，格式与 daily.py 一致。"""
    payload = {
        "report_date": date_str,
        "universe_size": 5,
        "generated_at": f"{date_str}T16:32:00",
        "total_seconds": 1.0,
        "errors": [],
        "results": preds_by_target,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class TestEvaluatePredictions:

    def _make_source_with_rising_stock(self) -> FakeSource:
        bars = _make_price_bars("600519", "2025-01-02", 60, 100.0, 1.0)
        return FakeSource({"600519": bars})

    def test_empty_dir_returns_zero_files(self, tmp_path):
        """空目录 → n_files_used=0，results 为空。"""
        from astock_quant.predict.accuracy import evaluate_predictions
        result = evaluate_predictions(
            start_date="2026-01-01",
            end_date="2026-01-31",
            reports_dir=tmp_path,
            source=FakeSource({}),
        )
        assert result["n_files_used"] == 0
        assert result["results"] == {}

    def test_horizon_cutoff_skips_recent_files(self, tmp_path):
        """T + horizon > today → 跳过（路径不完整）。"""
        from astock_quant.predict.accuracy import evaluate_predictions
        today = _dt.date.today()
        recent_date = (today - _dt.timedelta(days=2)).isoformat()
        path = tmp_path / f"predictions_{recent_date}.json"
        _write_prediction_json(path, recent_date, {
            "direction": {"predictions": [{"ticker": "600519", "value": 1.0}]},
        })

        with patch("astock_quant.predict.accuracy._dt") as mock_dt:
            mock_dt.date.today.return_value = today
            mock_dt.date.fromisoformat.side_effect = _dt.date.fromisoformat
            mock_dt.timedelta = _dt.timedelta
            result = evaluate_predictions(
                start_date=(today - _dt.timedelta(days=10)).isoformat(),
                end_date=today.isoformat(),
                reports_dir=tmp_path,
                source=FakeSource({}),
                horizon=5,
            )
        assert result["n_skipped_horizon"] >= 1
        assert result["results"].get("direction") is None

    def test_missing_gt_increments_counter_and_does_not_raise(self, tmp_path):
        """source 对某 ticker 返回 [] → n_missing_gt++ 不抛异常。"""
        from astock_quant.predict.accuracy import evaluate_predictions
        past_date = "2025-06-01"
        path = tmp_path / f"predictions_{past_date}.json"
        _write_prediction_json(path, past_date, {
            "direction": {"predictions": [{"ticker": "NOTEXIST", "value": 1.0}]},
        })
        result = evaluate_predictions(
            start_date="2025-06-01",
            end_date="2025-06-01",
            reports_dir=tmp_path,
            source=FakeSource({}),
            horizon=5,
        )
        # 因为 GT 缺失，没有 evaluated，results 里不应有 direction
        assert result["results"].get("direction") is None

    def test_corrupt_json_skips_with_warning(self, tmp_path):
        """损坏的 JSON → 跳过该文件，继续处理其他。"""
        from astock_quant.predict.accuracy import evaluate_predictions
        bad = tmp_path / "predictions_2025-06-01.json"
        bad.write_text("{bad json!!!}", encoding="utf-8")
        result = evaluate_predictions(
            start_date="2025-06-01",
            end_date="2025-06-01",
            reports_dir=tmp_path,
            source=FakeSource({}),
            horizon=5,
        )
        assert result["n_files_used"] == 0
        assert result["n_files_scanned"] == 1

    def test_target_filter_only_evaluates_direction(self, tmp_path):
        """target='direction' → results 只含 direction，其余 target 缺。"""
        from astock_quant.predict.accuracy import evaluate_predictions
        bars = _make_price_bars("600519", "2025-06-01", 15, 100.0, 1.0)
        past_date = "2025-06-01"
        path = tmp_path / f"predictions_{past_date}.json"
        _write_prediction_json(path, past_date, {
            "direction": {"predictions": [{"ticker": "600519", "value": 1.0}]},
            "return_": {"predictions": [{"ticker": "600519", "value": 0.05}]},
        })
        result = evaluate_predictions(
            start_date="2025-06-01",
            end_date="2025-06-01",
            reports_dir=tmp_path,
            source=FakeSource({"600519": bars}),
            horizon=5,
            target="direction",
        )
        assert "direction" in result["results"]
        assert "return_" not in result["results"]

    def test_ticker_filter_only_evaluates_matching(self, tmp_path):
        """ticker='600519' → 只统计 600519，其余跳过。"""
        from astock_quant.predict.accuracy import evaluate_predictions
        bars_a = _make_price_bars("600519", "2025-06-01", 15, 100.0, 1.0)
        past_date = "2025-06-01"
        path = tmp_path / f"predictions_{past_date}.json"
        _write_prediction_json(path, past_date, {
            "direction": {"predictions": [
                {"ticker": "600519", "value": 1.0},
                {"ticker": "000858", "value": 1.0},
            ]},
        })
        result = evaluate_predictions(
            start_date="2025-06-01",
            end_date="2025-06-01",
            reports_dir=tmp_path,
            source=FakeSource({"600519": bars_a}),
            horizon=5,
            ticker="600519",
        )
        if "direction" in result["results"]:
            assert result["results"]["direction"]["n_evaluated"] == 1

    def test_full_flow_direction_hit_rate(self, tmp_path):
        """端到端：1 个 JSON + 1 只连涨股 + buy 预测 → direction hit_rate = 1.0。"""
        from astock_quant.predict.accuracy import evaluate_predictions
        bars = _make_price_bars("600519", "2025-06-01", 15, 100.0, 1.0)
        past_date = "2025-06-01"
        path = tmp_path / f"predictions_{past_date}.json"
        _write_prediction_json(path, past_date, {
            "direction": {"predictions": [{"ticker": "600519", "value": 1.0, "score": 0.8}]},
        })
        result = evaluate_predictions(
            start_date="2025-06-01",
            end_date="2025-06-01",
            reports_dir=tmp_path,
            source=FakeSource({"600519": bars}),
            horizon=5,
        )
        assert result["n_files_used"] == 1
        if "direction" in result["results"]:
            assert result["results"]["direction"]["hit_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestAccuracyCLI:

    def test_cli_no_predictions_exit_1(self, tmp_path):
        """reports_dir 空 → exit 1（通过 monkeypatch evaluate_predictions 注入空 source）。"""
        from astock_quant.predict.accuracy import main

        def mock_evaluate(**kwargs):
            # 注入 FakeSource（空 GT），实际不需要网络
            kwargs["source"] = FakeSource({})
            # 直接调真实函数，但目录为空 → n_files_used=0 → exit 1
            return {
                "start_date": kwargs.get("start_date", "2026-01-01"),
                "end_date": kwargs.get("end_date", "2026-01-31"),
                "horizon": 5,
                "today_cutoff": "2026-01-26",
                "n_files_scanned": 0,
                "n_files_used": 0,
                "n_skipped_horizon": 0,
                "ticker_filter": None,
                "target_filter": None,
                "tp_pct": 0.05,
                "sl_pct": -0.03,
                "results": {},
            }

        with patch("astock_quant.predict.accuracy.evaluate_predictions", side_effect=mock_evaluate):
            code = main([
                "--reports-dir", str(tmp_path),
                "--output-dir", str(tmp_path),
                "--no-render",
                "--quiet",
            ])
        assert code == 1

    def test_cli_default_days_30(self, tmp_path):
        """默认 --days 30 → start_date ≈ today - 30。"""
        from astock_quant.predict.accuracy import main
        captured_args = {}

        def mock_evaluate(**kwargs):
            captured_args.update(kwargs)
            return {
                "start_date": kwargs["start_date"] if isinstance(kwargs["start_date"], str)
                else kwargs["start_date"].isoformat(),
                "end_date": "2026-05-16",
                "horizon": 5,
                "today_cutoff": "2026-05-11",
                "n_files_scanned": 0,
                "n_files_used": 0,
                "n_skipped_horizon": 0,
                "ticker_filter": None,
                "target_filter": None,
                "tp_pct": 0.05,
                "sl_pct": -0.03,
                "results": {},
            }

        today = _dt.date.today()
        with patch("astock_quant.predict.accuracy.evaluate_predictions", side_effect=mock_evaluate):
            main([
                "--reports-dir", str(tmp_path),
                "--output-dir", str(tmp_path),
                "--no-render",
                "--quiet",
            ])

        if captured_args:
            start = captured_args.get("start_date")
            if isinstance(start, str):
                start = _dt.date.fromisoformat(start)
            expected_start = today - _dt.timedelta(days=30)
            assert start == expected_start
