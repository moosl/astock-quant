"""东财分红接口限速与重试测试。"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import call, patch

import pandas as pd

from astock_quant.data.fundamentals import _fetch_dividends


def test_dividend_request_retries_with_backoff_then_succeeds():
    attempts = [RuntimeError("limited"), RuntimeError("limited"), pd.DataFrame({
        "报告期": ["2025-12-31"],
        "现金分红-现金分红比例": [10.0],
    })]

    def fetch(*, symbol):
        result = attempts.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    fake_akshare = SimpleNamespace(stock_fhps_detail_em=fetch)
    with patch.dict(sys.modules, {"akshare": fake_akshare}), \
         patch("astock_quant.data.fundamentals.time.sleep") as sleep:
        result = _fetch_dividends("600519")

    assert result == {"20251231": 1.0}
    assert sleep.call_args_list == [call(0.5), call(1.0), call(0.5), call(2.0), call(0.5)]


def test_dividend_request_stops_after_three_failures(caplog):
    calls = []

    def fetch(*, symbol):
        calls.append(symbol)
        raise TypeError("'NoneType' object is not subscriptable")

    fake_akshare = SimpleNamespace(stock_fhps_detail_em=fetch)
    with patch.dict(sys.modules, {"akshare": fake_akshare}), \
         patch("astock_quant.data.fundamentals.time.sleep") as sleep:
        result = _fetch_dividends("688047")

    assert result == {}
    assert calls == ["688047", "688047", "688047"]
    assert sleep.call_args_list == [call(0.5), call(1.0), call(0.5), call(2.0), call(0.5)]
    assert "已重试 3 次" in caplog.text
