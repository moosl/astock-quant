"""④ 买卖点信号扩展 端到端编排 —— P11 实装.

复用 run_direction / run_return / run_ranking 的 7 步骨架，是 ④ 买卖点信号的完整纵向切片。
一个函数串起：

    data.prepare_stage1_data()
        ↓
    factors.compute_factor_frame()      # FactorFrame（与 ① ② ③ 共用，0 改动）
        ↓
    labels.trade_signal_label()         # TP=+1 / HOLD=0 / SL=-1 三类标签
        ↓
    labels.align_xy()                   # 对齐成 (X, y)
        ↓
    models.time_series_split()          # 时序切分 + purge gap（命门，与 ① ② ③ 共用）
        ↓
    models.TradeSignalModel.fit/predict # LightGBM 3 类分类器
        ↓
    3 类 accuracy / per-class P/R/F1 / macro-F1 + 类别分布
        ↓
    BacktestEngine.run(buy_predictions) # 只喂 value=+1（TP 预测）给引擎
        ↓
    SignalGenerator.generate(predictions)   # trade_signal 分支

⑤ 止损止盈引擎触发（P11 §5.3 收盘价触发）目前由 BacktestEngine 现有逻辑处理，
run_trade_signal 在 pipeline 层 filter 预测，不改引擎。

用法：
    from astock_quant.pipeline.run_trade_signal import run_trade_signal
    result = run_trade_signal()
    print(result["metrics"])
    # → {'accuracy': 0.xx, 'macro_f1': 0.xx, 'tp_precision': 0.xx, ...}
    result["predictions"]        # list[Prediction]，全量验证集预测（含 TP/HOLD/SL）
    result["buy_predictions"]    # list[Prediction]，只含 value=+1（TP）的预测，用于回测
    result["backtest_metrics"]   # 回测指标
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from astock_quant.backtest.engine import BacktestEngine, BacktestRunConfig
from astock_quant.config.settings import SETTINGS
from astock_quant.contracts import BacktestResult, Prediction, SignalReport
from astock_quant.data.dataset import prepare_stage1_data
from astock_quant.factors.registry import compute_factor_frame
from astock_quant.labels.targets import align_xy, trade_signal_label
from astock_quant.models.splits import TimeSeriesSplit, time_series_split
from astock_quant.models.trade_signal import TradeSignalModel
from astock_quant.signals.generator import SignalGenerator

logger = logging.getLogger(__name__)


def run_trade_signal(
    *,
    universe: list[str] | None = None,
    tp_pct: float = 0.05,
    sl_pct: float = -0.03,
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
    predict_only: bool = False,
    predict_model_path: str | None = None,
    predict_date: str | None = None,
    prepared_data: dict | None = None,
) -> dict[str, Any]:
    """④ 买卖点信号扩展 完整端到端 —— 训练 + 验证 + 回测 + 信号.

    参数（全部可选；缺省走 SETTINGS）：
        universe:               股票池（同 run_direction H3 修复，已真接通）。
        tp_pct:                 止盈幅度（正数），默认 +5%。触发 label=+1（TP）。
        sl_pct:                 止损幅度（负数），默认 -3%。触发 label=-1（SL）。
                                必须 sl_pct < tp_pct，否则 trade_signal_label 抛 ValueError。
        train_end:              训练截止日，默认 SETTINGS.split.train_end。
        valid_end:              验证截止日，默认 SETTINGS.split.valid_end。
        horizon:                label 未来窗口（交易日），默认 SETTINGS.label.horizon (5)。
        purge_gap_days:         切分时的 purge gap，默认 SETTINGS.split.purge_gap (10)。
        model_params:           LightGBM 超参覆盖（dict），merge 进 TradeSignalModel.DEFAULT_PARAMS。
        early_stopping_rounds:  默认 30；传 None 关闭。
        save_model_to:          产物保存路径（如 "artifacts/trade_signal_lgbm.txt"）。None 不保存。
        force_refresh_data:     数据缓存强制重拉。默认 False（用缓存）。
        verbose:                True 时打印阶段耗时与中间统计。
        run_backtest:           True 时跑回测（默认 True）。
        backtest_config:        回测配置；None 走默认。
        predict_only:           **P12 新增**：True 时跳过 fit + valid metrics + 回测，从磁盘
                                加载已训练模型，对全 (date, ticker) 做 inference。供每日预测报告用。
                                返回 dict 含 `buy_predictions`（filter value=+1）方便上层组装。
        predict_model_path:     `predict_only=True` 时显式指定模型路径。None 则按
                                `artifacts/models/trade_signal_{predict_date}.lgb` 解析。
        predict_date:           `predict_only=True` 时的预测日期（YYYY-MM-DD），None 走今天。

    返回 dict：
        - predictions:        list[Prediction]，验证集全量预测（target_type="trade_signal"，含 TP/HOLD/SL）
        - buy_predictions:    list[Prediction]，只含 value=+1（TP 预测），用于回测引擎
        - metrics:            { accuracy, macro_f1, tp_precision, tp_recall, tp_f1,
                                hold_precision, hold_recall, hold_f1,
                                sl_precision, sl_recall, sl_f1,
                                n_tp, n_hold, n_sl, train_size, valid_size, ... }
        - factor_names:       训练用到的因子列名
        - feature_importance: pd.Series（gain 重要性，已排序）
        - split:              TimeSeriesSplit 对象
        - model:              TradeSignalModel（训练好的实例）
        - score_frame:        DataFrame[value, score, proba_sl, proba_hold, proba_tp]
        - y_valid:            真实 y_va（-1/0/+1 三类标签）
        - backtest:           BacktestResult（如果 run_backtest=True）
        - backtest_metrics:   回测指标 dict
        - signals:            SignalReport（trade_signal 分派）
    """
    t_total = time.time()
    horizon = horizon if horizon is not None else SETTINGS.label.horizon
    purge_gap_days = (
        purge_gap_days if purge_gap_days is not None else SETTINGS.split.purge_gap
    )

    # ---------- predict_only 模式短路 ----------
    if predict_only:
        return _run_trade_signal_predict_only(
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
    source = data["source"]
    if verbose:
        logger.info(
            "[1/7] data: prices=%s, moneyflow=%s, financials=%d tickers (%.2fs)",
            tuple(price_panel.shape),
            tuple(mf_panel.shape) if mf_panel is not None else "—",
            len(financials), time.time() - t0,
        )

    # ---------- 2. 因子（与 ① ② ③ 共用 compute_factor_frame，0 改动）----------
    t0 = time.time()
    ff = compute_factor_frame(
        price_panel=price_panel,
        moneyflow_panel=mf_panel,
        financials=financials,
        news_fetcher=source.get_news,
    )
    if verbose:
        logger.info(
            "[2/7] factors: shape=%s, %d factors (%.2fs)",
            tuple(ff.shape), len(ff.factor_names), time.time() - t0,
        )

    # ---------- 3. 标签 + 对齐（trade_signal_label：TP=+1 / HOLD=0 / SL=-1）----------
    t0 = time.time()
    y_series = trade_signal_label(
        price_panel,
        horizon=horizon,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        for_training=True,
    )
    X, y = align_xy(
        ff.data, y_series,
        drop_label_nan=True, drop_all_nan_rows=True,
    )
    if verbose:
        label_counts = {int(k): int(v) for k, v in y.value_counts().sort_index().items()}
        logger.info(
            "[3/7] labels+align: X=%s, y=%s, dist SL/HOLD/TP=%s (%.2fs)",
            tuple(X.shape), tuple(y.shape), label_counts, time.time() - t0,
        )

    # ---------- 4. 切分（命门，与 ① ② 共用 purge gap）----------
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

    # ---------- 5. 训练 + 预测 + 评估（3 类分类指标）----------
    t0 = time.time()
    model = TradeSignalModel(**(model_params or {}))
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=early_stopping_rounds,
    )
    score_frame = model.predict_score_frame(X_va)
    predictions: list[Prediction] = model.predict(X_va)

    # —— 3 类分类 metrics
    y_va_arr = y_va.values.astype(int)
    pred_labels = score_frame["value"].values.astype(int)

    accuracy = float(np.mean(y_va_arr == pred_labels))
    p_sl, r_sl, f1_sl = _prf(y_va_arr, pred_labels, label=-1)
    p_hold, r_hold, f1_hold = _prf(y_va_arr, pred_labels, label=0)
    p_tp, r_tp, f1_tp = _prf(y_va_arr, pred_labels, label=1)
    macro_f1 = float(np.mean([f1_sl, f1_hold, f1_tp]))

    n_sl = int((y_va_arr == -1).sum())
    n_hold = int((y_va_arr == 0).sum())
    n_tp = int((y_va_arr == 1).sum())

    metrics: dict[str, Any] = {
        "train_size": int(split.train_size),
        "valid_size": int(split.valid_size),
        "train_end": split.train_end.date().isoformat(),
        "valid_start": split.valid_start.date().isoformat(),
        "gap_days": int(split.gap_days),
        "label_horizon": int(split.label_horizon),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "sl_precision": p_sl,
        "sl_recall": r_sl,
        "sl_f1": f1_sl,
        "hold_precision": p_hold,
        "hold_recall": r_hold,
        "hold_f1": f1_hold,
        "tp_precision": p_tp,
        "tp_recall": r_tp,
        "tp_f1": f1_tp,
        "n_sl": n_sl,
        "n_hold": n_hold,
        "n_tp": n_tp,
        "n_features": int(X.shape[1]),
        "train_seconds": float(time.time() - t0),
    }
    if verbose:
        logger.info(
            "[5/7] train+eval: acc=%.4f, macro_F1=%.4f, "
            "TP(p=%.3f r=%.3f f1=%.3f) HOLD(p=%.3f r=%.3f f1=%.3f) SL(p=%.3f r=%.3f f1=%.3f) "
            "dist SL=%d HOLD=%d TP=%d (%.2fs)",
            accuracy, macro_f1,
            p_tp, r_tp, f1_tp, p_hold, r_hold, f1_hold, p_sl, r_sl, f1_sl,
            n_sl, n_hold, n_tp, metrics["train_seconds"],
        )

    if save_model_to:
        model.save(save_model_to)
        # Bug 1 修复：落 sidecar metadata.json
        _save_train_metadata(save_model_to, metrics)
        if verbose:
            logger.info("model + metadata saved → %s", save_model_to)

    # ---------- 6. 回测 + 7. 信号 ----------
    # trade_signal 的 score 表征「模型置信度」，不区分 TP/SL/HOLD 方向。
    # pipeline 层 filter 只保留 value=+1（TP 预测）喂给回测引擎，
    # 引擎按 score 排序取 Top-K —— 此时 score 高 = 对"会涨"更有把握。
    buy_predictions = [p for p in predictions if p.value == 1.0]

    backtest_result: BacktestResult | None = None
    signal_report: SignalReport | None = None
    if run_backtest:
        t0 = time.time()
        bt_cfg = backtest_config or BacktestRunConfig(
            initial_cash=SETTINGS.backtest.initial_capital,
            commission_rate=SETTINGS.backtest.commission_rate,
            stamp_tax_rate=SETTINGS.backtest.stamp_tax_rate,
        )
        engine = BacktestEngine(price_panel=price_panel, config=bt_cfg)
        backtest_result = engine.run(buy_predictions)
        if verbose:
            logger.info(
                "[6/7] backtest (buy_only): %d 个交易日，%d 笔交易，total_return=%s, sharpe=%s (%.2fs)",
                backtest_result.metrics.get("trading_days", 0),
                backtest_result.metrics.get("n_trades", 0),
                _fmt_pct(backtest_result.metrics.get("total_return")),
                _fmt_num(backtest_result.metrics.get("sharpe")),
                time.time() - t0,
            )

        t0 = time.time()
        gen = SignalGenerator()
        signal_report = gen.generate(predictions)
        if verbose:
            logger.info("[7/7] signals: %s (%.2fs)", signal_report.notes, time.time() - t0)

    metrics["total_seconds"] = float(time.time() - t_total)

    result: dict[str, Any] = {
        "predictions": predictions,
        "buy_predictions": buy_predictions,
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
DEFAULT_MODEL_TYPE_TRADE_SIGNAL = "trade_signal"

_TRAIN_METADATA_KEYS_TRADE_SIGNAL = {
    "train_size", "valid_size", "train_end", "valid_start",
    "gap_days", "label_horizon", "tp_pct", "sl_pct",
    "accuracy", "macro_f1",
    "sl_precision", "sl_recall", "sl_f1",
    "hold_precision", "hold_recall", "hold_f1",
    "tp_precision", "tp_recall", "tp_f1",
    "n_sl", "n_hold", "n_tp",
    "n_features", "train_seconds",
}


def _metadata_path(model_path: Path | str) -> Path:
    p = Path(model_path)
    return p.with_suffix(p.suffix + ".metadata.json")


def _save_train_metadata(model_path: Path | str, metrics: dict[str, Any]) -> None:
    """Bug 1 修复：训练 metrics 落 sidecar JSON."""
    import json as _json
    keep = {k: v for k, v in metrics.items() if k in _TRAIN_METADATA_KEYS_TRADE_SIGNAL}
    keep["_saved_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    keep["_model_type"] = DEFAULT_MODEL_TYPE_TRADE_SIGNAL
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
) -> Path:
    """解析 predict_only 模式下要加载的模型文件路径（与其他 pipeline 同款）."""
    if predict_model_path is not None:
        path = Path(predict_model_path)
    else:
        date_str = predict_date or _dt.date.today().isoformat()
        path = DEFAULT_MODELS_DIR / f"{model_type}_{date_str}.lgb"
    if not path.exists():
        raise FileNotFoundError(
            f"predict_only 模式找不到模型文件: {path}\n"
            f"请先跑一次训练并 save 到该路径：\n"
            f"  uv run python -c \"from astock_quant.pipeline.run_trade_signal import run_trade_signal; "
            f"run_trade_signal(save_model_to='{path}')\""
        )
    return path


def _run_trade_signal_predict_only(
    *,
    universe: list[str] | None,
    force_refresh_data: bool,
    verbose: bool,
    predict_model_path: str | None,
    predict_date: str | None,
    t_total: float,
    prepared_data: dict | None = None,
) -> dict[str, Any]:
    """predict_only 模式：load 模型 + 拉数据 + 算因子 + 全 (date, ticker) 3 类 predict.

    与其他 pipeline 的差异：3 类输出，额外提供 `buy_predictions`（filter value=+1）
    给上层用。
    """
    model_path = _resolve_predict_model_path(
        DEFAULT_MODEL_TYPE_TRADE_SIGNAL, predict_model_path, predict_date,
    )
    if verbose:
        logger.info("[predict_only] loading model: %s", model_path)
    model = TradeSignalModel().load(model_path)

    t0 = time.time()
    if prepared_data is not None:
        data = prepared_data
    else:
        data = prepare_stage1_data(universe=universe, force_refresh=force_refresh_data)
    price_panel = data["prices"]
    mf_panel = data["moneyflow"]
    financials = data["financials"]
    source = data["source"]
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
        news_fetcher=source.get_news,
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
    buy_predictions = [p for p in predictions if p.value == 1.0]
    if verbose:
        logger.info(
            "[predict_only 3/3] inference: %d predictions (TP=%d) over X=%s (latest day only) (%.2fs)",
            len(predictions), len(buy_predictions), tuple(X_inf.shape), time.time() - t0,
        )

    train_metadata = _load_train_metadata(model_path)

    metrics: dict[str, Any] = {
        "mode": "predict_only",
        "model_path": str(model_path),
        "n_predictions": len(predictions),
        "n_tp_predictions": len(buy_predictions),
        "total_seconds": float(time.time() - t_total),
    }
    if train_metadata:
        for k, v in train_metadata.items():
            if k not in metrics:
                metrics[k] = v

    return {
        "predictions": predictions,
        "buy_predictions": buy_predictions,
        "score_frame": score_frame,
        "model": model,
        "predict_model_path": str(model_path),
        "metrics": metrics,
        "factor_names": ff.factor_names,
    }


def _prf(y_true: np.ndarray, y_pred: np.ndarray, *, label: int) -> tuple[float, float, float]:
    """单类 precision / recall / F1."""
    tp = int(((y_pred == label) & (y_true == label)).sum())
    fp = int(((y_pred == label) & (y_true != label)).sum())
    fn = int(((y_pred != label) & (y_true == label)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


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


__all__ = ["run_trade_signal"]
