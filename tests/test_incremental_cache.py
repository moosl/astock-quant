"""行情增量缓存与财务缓存刷新策略测试。"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from astock_quant.contracts import FinancialMetrics, PriceBar
from astock_quant.config.settings import Settings
from astock_quant.data import cache, fundamentals
from astock_quant.data.dataset import load_prices


class FakePriceSource:
    def __init__(self, bars: list[PriceBar] | None = None):
        self.bars = bars or []
        self.calls: list[tuple[str, str, str]] = []

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[PriceBar]:
        self.calls.append((ticker, start_date, end_date))
        return self.bars


def _bar(day: str, close: float = 10.0) -> PriceBar:
    return PriceBar(
        ticker="600519",
        date=date.fromisoformat(day),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100.0,
    )


def _cached_history(last_day: str) -> pd.DataFrame:
    return pd.DataFrame([
        _bar("2022-01-01").model_dump(),
        _bar(last_day).model_dump(),
    ])


def _use_cache_dir(tmp_path, monkeypatch) -> None:
    settings = SimpleNamespace(data_cache_dir=tmp_path)
    monkeypatch.setattr(cache, "SETTINGS", settings)
    monkeypatch.setattr(fundamentals, "SETTINGS", settings)


def test_default_history_end_is_today():
    assert Settings().history_end == date.today().isoformat()


def test_fresh_price_cache_skips_network(tmp_path, monkeypatch):
    _use_cache_dir(tmp_path, monkeypatch)
    cache.write_cache(_cached_history("2026-07-15"), "prices", "600519")
    source = FakePriceSource()

    result = load_prices("600519", source, "2022-01-01", "2026-07-16")

    assert source.calls == []
    assert result["date"].max() == pd.Timestamp("2026-07-15")


def test_fresh_but_outdated_price_cache_still_updates(tmp_path, monkeypatch):
    _use_cache_dir(tmp_path, monkeypatch)
    cache.write_cache(_cached_history("2026-05-15"), "prices", "600519")
    source = FakePriceSource([_bar("2026-07-16")])

    result = load_prices("600519", source, "2022-01-01", "2026-07-16")

    assert source.calls == [("600519", "2026-05-16", "2026-07-16")]
    assert result["date"].max() == pd.Timestamp("2026-07-16")


def test_stale_price_cache_only_fetches_missing_tail(tmp_path, monkeypatch):
    _use_cache_dir(tmp_path, monkeypatch)
    cache.write_cache(_cached_history("2026-07-14"), "prices", "600519")
    source = FakePriceSource([_bar("2026-07-15"), _bar("2026-07-16")])

    with patch("astock_quant.data.dataset.cache.is_fresh", return_value=False):
        result = load_prices("600519", source, "2022-01-01", "2026-07-16")

    assert source.calls == [("600519", "2026-07-15", "2026-07-16")]
    assert result["date"].min() == pd.Timestamp("2022-01-01")
    assert result["date"].max() == pd.Timestamp("2026-07-16")


def test_incremental_price_merge_deduplicates_rows(tmp_path, monkeypatch):
    _use_cache_dir(tmp_path, monkeypatch)
    cache.write_cache(_cached_history("2026-07-14"), "prices", "600519")
    source = FakePriceSource([_bar("2026-07-14", 11.0), _bar("2026-07-15")])

    with patch("astock_quant.data.dataset.cache.is_fresh", return_value=False):
        result = load_prices("600519", source, "2022-01-01", "2026-07-15")

    assert len(result) == 3
    assert result.loc[result["date"] == pd.Timestamp("2026-07-14"), "close"].item() == 11.0


def test_price_refresh_failure_falls_back_to_stale_cache(tmp_path, monkeypatch):
    _use_cache_dir(tmp_path, monkeypatch)
    cache.write_cache(_cached_history("2026-07-14"), "prices", "600519")
    source = FakePriceSource()

    with patch("astock_quant.data.dataset.cache.is_fresh", return_value=False):
        result = load_prices("600519", source, "2022-01-01", "2026-07-16")

    assert source.calls == [("600519", "2026-07-15", "2026-07-16")]
    assert result["date"].max() == pd.Timestamp("2026-07-14")


def test_fresh_financial_cache_skips_network(tmp_path, monkeypatch):
    _use_cache_dir(tmp_path, monkeypatch)
    cached = FinancialMetrics(ticker="600519", report_period="20251231")
    fundamentals.write_cache([cached], "600519")

    with patch("astock_quant.data.fundamentals.fetch_financial_history") as fetch:
        result = fundamentals.load_financial_history("600519")

    fetch.assert_not_called()
    assert [record.report_period for record in result] == ["20251231"]


def test_stale_financial_cache_refreshes_and_failure_falls_back(tmp_path, monkeypatch):
    _use_cache_dir(tmp_path, monkeypatch)
    cached = FinancialMetrics(ticker="600519", report_period="20251231")
    fundamentals.write_cache([cached], "600519")

    with patch("astock_quant.data.fundamentals.data_cache.is_fresh", return_value=False), \
         patch("astock_quant.data.fundamentals.fetch_financial_history", return_value=[]):
        result = fundamentals.load_financial_history("600519")

    assert [record.report_period for record in result] == ["20251231"]


def test_stale_financial_cache_is_replaced_after_success(tmp_path, monkeypatch):
    _use_cache_dir(tmp_path, monkeypatch)
    fundamentals.write_cache(
        [FinancialMetrics(ticker="600519", report_period="20251231")],
        "600519",
    )
    refreshed = FinancialMetrics(ticker="600519", report_period="20260331")

    with patch("astock_quant.data.fundamentals.data_cache.is_fresh", return_value=False), \
         patch(
             "astock_quant.data.fundamentals.fetch_financial_history",
             return_value=[refreshed],
         ):
        result = fundamentals.load_financial_history("600519")

    assert [record.report_period for record in result] == ["20260331"]
    assert [record.report_period for record in fundamentals.read_cache("600519")] == ["20260331"]
