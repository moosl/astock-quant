"""每日预测报告包 —— Stage 4 P12.

入口：
    from astock_quant.predict.daily import run_daily_predict, main
    或 CLI: `uv run python -m astock_quant.predict.daily --date today`

依赖 4 个 pipeline 已训练并 save 到 `artifacts/models/{type}_{date}.lgb`。

不在包级别 eager-import `daily.py`，避免 `python -m astock_quant.predict.daily`
触发的 「module 同时被作为包成员和脚本加载」RuntimeWarning。
按需 `from astock_quant.predict.daily import run_daily_predict` 即可。
"""

__all__ = ["run_daily_predict"]


def __getattr__(name: str):
    # Lazy attribute import：访问 `astock_quant.predict.run_daily_predict` 时才 import
    if name == "run_daily_predict":
        from astock_quant.predict.daily import run_daily_predict
        return run_daily_predict
    raise AttributeError(f"module 'astock_quant.predict' has no attribute {name!r}")
