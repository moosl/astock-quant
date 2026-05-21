"""② 收益率回归 端到端编排 —— P9 实装.

复用 run_direction.py 的 7 步骨架，是 ② 收益率回归的完整纵向切片。一个函数串起：

    data.prepare_stage1_data()
        ↓
    factors.compute_factor_frame()    # FactorFrame（与 ① 共用，0 改动）
        ↓
    labels.return_label()             # 连续 float 收益率 y（替换 direction_label）
        ↓
    labels.align_xy()                 # 对齐成 (X, y)
        ↓
    models.time_series_split()        # 时序切分 + purge gap（命门，与 ① 共用）
        ↓
    models.ReturnRegressor.fit/predict
        ↓
    RMSE / MAE / R² / IC / rank-IC + Prediction list
        ↓
    BacktestEngine.run(predictions)   # 回测（与 ① 共用 engine，0 改动）
        ↓
    SignalGenerator.generate(predictions)  # 信号（②分支已升级）

③④ 实现时各加 run_ranking / run_trade_signal，复用同一编排骨架，
只换 labels.* + models.* + signals 分派两个调用。

用法：
    from astock_quant.pipeline.run_return import run_return
    result = run_return()
    print(result["metrics"])
    # → {'train_size': ..., 'valid_size': ..., 'rmse': 0.0xx, 'mae': 0.0xx, 'r2': 0.0xx, 'ic': 0.0xx, ...}
    result["predictions"]   # list[Prediction]，对应验证集每一行
    result["backtest_metrics"]  # 回测指标
    result["signals"]       # SignalReport
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from astock_quant.backtest.engine import BacktestEngine, BacktestRunConfig
from astock_quant.config.settings import SETTINGS
from astock_quant.contracts import BacktestResult, Prediction, SignalReport
from astock_quant.data.dataset import prepare_stage1_data
from astock_quant.factors.registry import compute_factor_frame
from astock_quant.labels.targets import align_xy, return_label
from astock_quant.models.ret_regression import ReturnRegressor
from astock_quant.models.splits import TimeSeriesSplit, time_series_split
from astock_quant.signals.generator import SignalGenerator

logger = logging.getLogger(__name__)


def run_return(
    *,
    universe: list[str] | None = None,
    train_end: str | None = None,
    valid_end: str | None = None,
    horizon: int | None = None,
    purge_gap_days: int | None = None,
    model_params: dict[str, Any] | None = None,
    early_stopping_rounds: int | None = 30,
    save_model_to: str | None = None,
    force_refresh_data: bool = False,
    verbose: bool = True,
    run_backtest: bool = True,
    backtest_config: BacktestRunConfig | None = None,
    signal_buy_threshold: float = 0.02,
    signal_sell_threshold: float = -0.02,
    predict_only: bool = False,
    predict_model_path: str | None = None,
    predict_date: str | None = None,
    prepared_data: dict | None = None,
) -> dict[str, Any]:
    """② 收益率回归 完整端到端 —— 训练 + 验证 + 回测 + 信号.

    参数（全部可选；缺省走 SETTINGS）：
        universe:               股票池（与 run_direction H3 修复同款，已真接通）。
        train_end:              训练截止日，默认 SETTINGS.split.train_end。
        valid_end:              验证截止日，默认 SETTINGS.split.valid_end。
        horizon:                label 未来窗口（交易日），默认 SETTINGS.label.horizon (5)。
        purge_gap_days:         切分时的 purge gap，默认 SETTINGS.split.purge_gap (10)。
        model_params:           LightGBM 超参覆盖（dict），merge 进 ReturnRegressor.DEFAULT_PARAMS。
        early_stopping_rounds:  默认 30；传 None 关闭。
        save_model_to:          产物保存路径（如 "artifacts/return_lgbm.txt"）。None 不保存。
        force_refresh_data:     数据缓存强制重拉。默认 False（用缓存）。
        verbose:                True 时打印阶段耗时与中间统计。
        run_backtest:           True 时跑回测（默认 True）。
        backtest_config:        回测配置；None 走默认（含 H4 默认 hold）。
        signal_buy_threshold:   信号层 buy 阈值（预测收益率 ≥ 此值 → buy），默认 +2%。
        signal_sell_threshold:  信号层 sell 阈值（预测收益率 < 此值 → sell），默认 -2%。
        predict_only:           **P12 新增**：True 时跳过 fit + valid metrics + 回测，从磁盘
                                加载已训练模型，对全 (date, ticker) 做 inference。供每日预测报告用。
        predict_model_path:     `predict_only=True` 时显式指定模型路径。None 则按
                                `artifacts/models/return_{predict_date}.lgb` 解析。
                                文件不存在时抛 FileNotFoundError 给出复现命令。
        predict_date:           `predict_only=True` 时的预测日期（YYYY-MM-DD），None 走今天。

    返回 dict：
        - predictions:        list[Prediction]，验证集预测（target_type="return"）
        - metrics:            { train_size, valid_size, rmse, mae, r2, ic, rank_ic, ... }
        - factor_names:       训练用到的因子列名
        - feature_importance: pd.Series（gain 重要性，已排序）
        - split:              TimeSeriesSplit 对象
        - model:              ReturnRegressor（训练好的实例）
        - score_frame:        DataFrame[value, score]，索引与 X_va 一致
        - y_valid:            真实 y_va
        - backtest:           BacktestResult（如果 run_backtest=True）
        - backtest_metrics:   回测指标 dict
        - signals:            SignalReport（return 分派，含 buy/sell/hold + 强度）
    """
    t_total = time.time()
    horizon = horizon if horizon is not None else SETTINGS.label.horizon
    purge_gap_days = (
        purge_gap_days if purge_gap_days is not None else SETTINGS.split.purge_gap
    )

    # ---------- predict_only 模式短路 ----------
    if predict_only:
        return _run_return_predict_only(
            universe=universe,
            force_refresh_data=force_refresh_data,
            verbose=verbose,
            predict_model_path=predict_model_path,
            predict_date=predict_date,
            t_total=t_total,
            prepared_data=prepared_data,
        )

    # ---------- 1. 数据 ----------
    t0 = time.time()
    data = prepare_stage1_data(universe=universe, force_refresh=force_refresh_data)
    price_panel = data["prices"]
    mf_panel = data["moneyflow"]
    financials = data["financials"]
    if verbose:
        logger.info(
            "[1/7] data: prices=%s, moneyflow=%s, financials=%d tickers (%.2fs)",
            tuple(price_panel.shape),
            tuple(mf_panel.shape) if mf_panel is not None else "—",
            len(financials), time.time() - t0,
        )

    # ---------- 2. 因子（与 ① 共用 compute_factor_frame，0 改动）----------
    t0 = time.time()
    ff = compute_factor_frame(
        price_panel=price_panel,
        moneyflow_panel=mf_panel,
        financials=financials,
    )
    if verbose:
        logger.info(
            "[2/7] factors: shape=%s, %d factors (%.2fs)",
            tuple(ff.shape), len(ff.factor_names), time.time() - t0,
        )

    # ---------- 3. 标签 + 对齐（用 return_label，不二值化）----------
    t0 = time.time()
    y_series = return_label(price_panel, horizon=horizon, for_training=True)
    X, y = align_xy(
        ff.data, y_series,
        drop_label_nan=True, drop_all_nan_rows=True,
    )
    if verbose:
        logger.info(
            "[3/7] labels+align: X=%s, y=%s, y stats: mean=%.4f std=%.4f (%.2fs)",
            tuple(X.shape), tuple(y.shape), float(y.mean()), float(y.std()),
            time.time() - t0,
        )

    # ---------- 4. 切分（命门，与 ① 共用 splits 代码 + purge_gap）----------
    split: TimeSeriesSplit = time_series_split(
        X.index,
        train_end=train_end,
        valid_end=valid_end,
        purge_gap_days=purge_gap_days,
        label_horizon=horizon,
    )
    if verbose:
        logger.info("[4/7] split: %s", split.summary())

    X_tr, y_tr = X.loc[split.train_mask], y.loc[split.train_mask]
    X_va, y_va = X.loc[split.valid_mask], y.loc[split.valid_mask]
    if X_tr.empty or X_va.empty:
        raise RuntimeError(
            f"切分后 train 或 valid 为空：train={X_tr.shape}, valid={X_va.shape}。"
            "可能 train_end / valid_end 与数据区间不匹配。"
        )

    # ---------- 5. 训练 + 预测 + 评估（回归指标，区别于 ① 的分类指标）----------
    t0 = time.time()
    model = ReturnRegressor(**(model_params or {}))
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=early_stopping_rounds,
    )
    score_frame = model.predict_score_frame(X_va)
    predictions: list[Prediction] = model.predict(X_va)

    # —— 回归 metrics
    y_va_arr = y_va.values.astype(float)
    pred_arr = score_frame["score"].values.astype(float)
    metrics = {
        "train_size": int(split.train_size),
        "valid_size": int(split.valid_size),
        "train_end": split.train_end.date().isoformat(),
        "valid_start": split.valid_start.date().isoformat(),
        "gap_days": int(split.gap_days),
        "label_horizon": int(split.label_horizon),
        "y_train_mean": float(y_tr.mean()),
        "y_train_std": float(y_tr.std()),
        "y_valid_mean": float(y_va.mean()),
        "y_valid_std": float(y_va.std()),
        "rmse": float(np.sqrt(mean_squared_error(y_va_arr, pred_arr))),
        "mae": float(mean_absolute_error(y_va_arr, pred_arr)),
        "r2": float(r2_score(y_va_arr, pred_arr)),
        "ic": _pearson_ic(y_va_arr, pred_arr),
        "rank_ic": _spearman_ic(y_va_arr, pred_arr),
        "n_features": int(X.shape[1]),
        "train_seconds": float(time.time() - t0),
    }
    if verbose:
        logger.info(
            "[5/7] train+eval: RMSE=%.4f, MAE=%.4f, R²=%.4f, IC=%.4f, rankIC=%.4f (%.2fs)",
            metrics["rmse"], metrics["mae"], metrics["r2"], metrics["ic"], metrics["rank_ic"],
            metrics["train_seconds"],
        )

    if save_model_to:
        model.save(save_model_to)
        # Bug 1 修复：落 sidecar metadata.json
        _save_train_metadata(save_model_to, metrics)
        if verbose:
            logger.info("model + metadata saved → %s", save_model_to)

    # ---------- 6. 回测 + 7. 信号 ----------
    backtest_result: BacktestResult | None = None
    signal_report: SignalReport | None = None
    if run_backtest:
        t0 = time.time()
        # 回测 config —— 与 run_direction 同款 SETTINGS 三字段填充
        bt_cfg = backtest_config or BacktestRunConfig(
            initial_cash=SETTINGS.backtest.initial_capital,
            commission_rate=SETTINGS.backtest.commission_rate,
            stamp_tax_rate=SETTINGS.backtest.stamp_tax_rate,
        )
        engine = BacktestEngine(price_panel=price_panel, config=bt_cfg)
        backtest_result = engine.run(predictions)
        if verbose:
            logger.info(
                "[6/7] backtest: %d 个交易日，%d 笔交易，total_return=%s, sharpe=%s (%.2fs)",
                backtest_result.metrics.get("trading_days", 0),
                backtest_result.metrics.get("n_trades", 0),
                _fmt_pct(backtest_result.metrics.get("total_return")),
                _fmt_num(backtest_result.metrics.get("sharpe")),
                time.time() - t0,
            )

        t0 = time.time()
        # 信号生成 —— SignalGenerator 的 return 分支按 buy/sell 阈值过滤
        gen = SignalGenerator(
            return_buy_threshold=signal_buy_threshold,
            return_sell_threshold=signal_sell_threshold,
        )
        signal_report = gen.generate(predictions)
        if verbose:
            logger.info("[7/7] signals: %s (%.2fs)", signal_report.notes, time.time() - t0)

    metrics["total_seconds"] = float(time.time() - t_total)

    result: dict[str, Any] = {
        "predictions": predictions,
        "metrics": metrics,
        "factor_names": ff.factor_names,
        "feature_importance": model.feature_importance(),
        "split": split,
        "model": model,
        "score_frame": score_frame,
        "y_valid": y_va,
    }
    if backtest_result is not None:
        result["backtest"] = backtest_result
        result["backtest_metrics"] = backtest_result.metrics
    if signal_report is not None:
        result["signals"] = signal_report
    return result


# ===========================================================================
# helpers
# ===========================================================================


DEFAULT_MODELS_DIR = Path("artifacts/models")
DEFAULT_MODEL_TYPE_RETURN = "return"

_TRAIN_METADATA_KEYS_RETURN = {
    "train_size", "valid_size", "train_end", "valid_start",
    "gap_days", "label_horizon",
    "y_train_mean", "y_train_std", "y_valid_mean", "y_valid_std",
    "rmse", "mae", "r2", "ic", "rank_ic",
    "n_features", "train_seconds",
}


def _metadata_path(model_path: Path | str) -> Path:
    p = Path(model_path)
    return p.with_suffix(p.suffix + ".metadata.json")


def _save_train_metadata(model_path: Path | str, metrics: dict[str, Any]) -> None:
    """Bug 1 修复：训练 metrics 落 sidecar JSON（白名单过滤）."""
    import json as _json
    keep = {k: v for k, v in metrics.items() if k in _TRAIN_METADATA_KEYS_RETURN}
    keep["_saved_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    keep["_model_type"] = DEFAULT_MODEL_TYPE_RETURN
    path = _metadata_path(model_path)
    with path.open("w", encoding="utf-8") as f:
        _json.dump(keep, f, ensure_ascii=False, indent=2)


def _load_train_metadata(model_path: Path | str) -> dict[str, Any]:
    import json as _json
    path = _metadata_path(model_path)
    if not path.exists():
        logger.warning("未找到训练 metadata sidecar: %s —— 诚信声明指标将显示 N/A", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return _json.load(f)
    except (OSError, _json.JSONDecodeError) as e:
        logger.warning("读取 metadata %s 失败: %s", path, e)
        return {}


def _filter_latest_day(factor_data, universe: list[str] | None):
    """Bug 2 修复：只挑「最新一天 × universe」的行喂模型."""
    if factor_data is None or factor_data.empty:
        return factor_data
    valid_mask = ~factor_data.isna().all(axis=1)
    df = factor_data.loc[valid_mask]
    if df.empty:
        return df
    dates = df.index.get_level_values("date")
    latest = dates.max()
    df = df[dates == latest]
    if universe:
        df = df[df.index.get_level_values("ticker").isin(set(universe))]
    return df


def _resolve_predict_model_path(
    model_type: str,
    predict_model_path: str | None,
    predict_date: str | None,
    models_dir: Path | None = None,
) -> Path:
    """解析 predict_only 模式下要加载的模型文件路径.

    与 run_direction._resolve_predict_model_path 同款逻辑（4 个 pipeline 各自 inline，
    避免引入新共享模块；未来稳定后可抽到 predict/_helpers.py）。

    P23 fix：当天模型不存在时 fallback 到最近一个 `{model_type}_*.lgb`，
    避免「当天没训练 → daily.py 报错 → 报告永远出不来」。
    """
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR

    if predict_model_path is not None:
        path = Path(predict_model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"predict_only 模式找不到显式指定的模型文件: {path}"
            )
        return path

    date_str = predict_date or _dt.date.today().isoformat()
    path = models_dir / f"{model_type}_{date_str}.lgb"
    if path.exists():
        return path

    candidates = sorted(models_dir.glob(f"{model_type}_*.lgb"))
    if candidates:
        fallback = candidates[-1]
        logger.warning(
            "predict_only：当天模型 %s 不存在，fallback 用最近的 %s",
            path.name, fallback.name,
        )
        return fallback

    raise FileNotFoundError(
        f"predict_only 模式找不到任何 {model_type} 模型文件："
        f"{models_dir}/{model_type}_*.lgb\n"
        f"请先跑一次训练并 save：\n"
        f"  uv run python -c \"from astock_quant.pipeline.run_return import run_return; "
        f"run_return(save_model_to='{path}')\""
    )


def _run_return_predict_only(
    *,
    universe: list[str] | None,
    force_refresh_data: bool,
    verbose: bool,
    predict_model_path: str | None,
    predict_date: str | None,
    t_total: float,
    prepared_data: dict | None = None,
) -> dict[str, Any]:
    """predict_only 模式：load 模型 + 拉数据 + 算因子 + 全 (date, ticker) predict.

    与训练路径的差异：跳过 labels + splits + fit + valid metrics + 回测 + 信号。
    """
    model_path = _resolve_predict_model_path(
        DEFAULT_MODEL_TYPE_RETURN, predict_model_path, predict_date,
    )
    if verbose:
        logger.info("[predict_only] loading model: %s", model_path)
    model = ReturnRegressor().load(model_path)

    t0 = time.time()
    if prepared_data is not None:
        data = prepared_data
    else:
        data = prepare_stage1_data(universe=universe, force_refresh=force_refresh_data)
    price_panel = data["prices"]
    mf_panel = data["moneyflow"]
    financials = data["financials"]
    if verbose:
        logger.info(
            "[predict_only 1/3] data: prices=%s (%.2fs)",
            tuple(price_panel.shape), time.time() - t0,
        )

    t0 = time.time()
    ff = compute_factor_frame(
        price_panel=price_panel,
        moneyflow_panel=mf_panel,
        financials=financials,
    )
    if verbose:
        logger.info(
            "[predict_only 2/3] factors: shape=%s (%.2fs)",
            tuple(ff.shape), time.time() - t0,
        )

    t0 = time.time()
    X_inf = _filter_latest_day(ff.data, universe)
    score_frame = model.predict_score_frame(X_inf)
    predictions: list[Prediction] = model.predict(X_inf)
    if verbose:
        logger.info(
            "[predict_only 3/3] inference: %d predictions over X=%s (latest day only) (%.2fs)",
            len(predictions), tuple(X_inf.shape), time.time() - t0,
        )

    train_metadata = _load_train_metadata(model_path)

    metrics: dict[str, Any] = {
        "mode": "predict_only",
        "model_path": str(model_path),
        "n_predictions": len(predictions),
        "total_seconds": float(time.time() - t_total),
    }
    if train_metadata:
        for k, v in train_metadata.items():
            if k not in metrics:
                metrics[k] = v

    return {
        "predictions": predictions,
        "score_frame": score_frame,
        "model": model,
        "predict_model_path": str(model_path),
        "metrics": metrics,
        "factor_names": ff.factor_names,
    }


def _pearson_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """信息系数 IC = Pearson correlation 介于真实 y 和预测 y.

    量化研究常用：IC > 0.05 算有效因子的弱信号；> 0.1 算相当强。
    全样本或常数预测时返回 NaN。
    """
    if len(y_true) < 2 or np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank IC = Spearman correlation —— 对预测大小不敏感、只看排序.

    量化研究里 rank-IC 是横截面选股的核心指标，更稳健（异常值不影响 rank）。
    """
    if len(y_true) < 2 or np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return float("nan")
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if rho is not None and not np.isnan(rho) else float("nan")


def _fmt_pct(v) -> str:
    if v is None:
        return "None"
    try:
        return f"{float(v) * 100:+.2f}%"
    except (TypeError, ValueError):
        return str(v)


def _fmt_num(v) -> str:
    if v is None:
        return "None"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


__all__ = ["run_return"]
