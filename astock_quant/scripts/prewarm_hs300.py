"""预热沪深 300 全量数据缓存.

一次性把 300 只成分股的 OHLC + 财务 + 资金流拉到 data_cache/，后续
pipeline 直接复用缓存，不再实时拉取。

用法：
    uv run python -m astock_quant.scripts.prewarm_hs300
"""

from __future__ import annotations

import sys
import time

from astock_quant.config.settings import get_hs300_universe
from astock_quant.data.astock_source import AStockSource, normalize_ticker
from astock_quant.data.dataset import load_financials, load_moneyflow, load_prices
from astock_quant.config.settings import SETTINGS


def main() -> None:
    universe = get_hs300_universe()
    source = AStockSource()
    total = len(universe)
    missing: list[str] = []

    print(f"开始预热沪深 300，共 {total} 只，区间 {SETTINGS.history_start} ~ {SETTINGS.history_end}", flush=True)
    t0 = time.time()

    for i, ticker in enumerate(universe, 1):
        code = normalize_ticker(ticker)
        print(f"[{i}/{total}] fetching {code} ...", flush=True)

        ok = True
        for attempt in range(3):
            try:
                load_prices(
                    code, source,
                    SETTINGS.history_start, SETTINGS.history_end,
                    force_refresh=False,
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  prices FAIL {code}: {e}", flush=True)
                    ok = False

        for attempt in range(3):
            try:
                load_moneyflow(
                    code, source,
                    SETTINGS.history_start, SETTINGS.history_end,
                    force_refresh=False,
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  moneyflow FAIL {code}: {e}", flush=True)

        for attempt in range(3):
            try:
                load_financials(universe=[code], source=source)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  financials FAIL {code}: {e}", flush=True)

        if not ok:
            missing.append(code)

    elapsed = time.time() - t0
    print(f"\n完成：{total - len(missing)}/{total} 只成功，{len(missing)} 只失败，耗时 {elapsed:.1f}s", flush=True)
    if missing:
        print(f"失败列表：{missing}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
