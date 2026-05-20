"""① 涨跌方向 端到端编排 —— Stage 1 的完整纵向切片.

这是 P1 计划「全流程一条命令可复现」的兑现。一个函数串起：

    data.prepare_stage1_data()
        ↓
    factors.compute_factor_frame()   # FactorFrame
        ↓
    labels.direction_label()         # direction y
        ↓
    labels.align_xy()                # 对齐成 (X, y)
        ↓
    models.time_series_split()       # 时序切分 + purge gap（命门）
        ↓
    models.DirectionModel.fit/predict
        ↓
    metrics + Prediction list

②③④ 实现时各加 run_return / run_ranking / run_trade_signal，复用同一编排骨架，
只换 labels.* + models.* 两个调用。

用法：
    from astock_quant.pipeline.run_direction import run_direction
    result = run_direction()
    print(result["metrics"])
    # → {'train_size': ..., 'valid_size': ..., 'accuracy': 0.xx, 'auc': 0.xx, 'log_loss': 0.xx, ...}
    result["predictions"]   # list[Prediction]，对应验证集每一行
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from astock_quant.backtest.engine import BacktestRunConfig, BacktestEngine
from astock_quant.config.settings import SETTINGS
from astock_quant.contracts import BacktestResult, Prediction, SignalReport
from astock_quant.data.dataset import prepare_stage1_data
from astock_quant.factors.registry import compute_factor_frame
from astock_quant.labels.targets import align_xy, direction_label
from astock_quant.models.direction import DirectionModel
from astock_quant.models.splits import TimeSeriesSplit, time_series_split
from astock_quant.signals.generator import SignalGenerator

logger = logging.getLogger(__name__)


def run_direction(
    *,
    universe: list[str] | None = None,
    train_end: str | None = None,
    valid_end: str | None = None,
    horizon: int | None = None,
    threshold: float | None = None,
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
    """① 涨跌方向 完整端到端 —— 训练 + 验证 + 关键指标.

    参数（全部可选；缺省走 SETTINGS）：
        universe:               股票池（默认 SETTINGS.universe，30 只蓝筹）。
                                P5 reviewer H3 修复：本参数已真接通 prepare_stage1_data ——
                                传入 list 即可换池子跑（Stage 2 常用「小池子先试 LLM、大池子上正式」）。
        train_end:              训练截止日，默认 SETTINGS.split.train_end ("2025-06-30")。
        valid_end:              验证截止日，默认 SETTINGS.split.valid_end ("2026-05-01")。
        horizon:                label 未来窗口（交易日），默认 SETTINGS.label.horizon (5)。
        threshold:              direction 阈值，默认 SETTINGS.label.direction_threshold (0.0)。
        purge_gap_days:         切分时的 purge gap（交易日），默认 SETTINGS.split.purge_gap (10)。
                                必须 >= horizon —— time_series_split 会校验。
        model_params:           LightGBM 超参覆盖（dict），merge 进 DirectionModel.DEFAULT_PARAMS。
        early_stopping_rounds:  默认 30；传 None 关闭。
        save_model_to:          产物保存路径（如 "artifacts/direction_lgbm.txt"）。None 不保存。
        force_refresh_data:     数据缓存强制重拉。默认 False（用缓存）。
        verbose:                True 时打印阶段耗时与中间统计。
        predict_only:           **P12 新增**：True 时跳过 fit + valid metrics，从磁盘加载已训练
                                模型，对**全部 (date, ticker)** 做 predict。供每日预测报告用。
                                不报训练 metrics（valid 集对每日预测无意义）；不跑回测。
        predict_model_path:     `predict_only=True` 时显式指定模型文件路径。None 则按
                                `artifacts/models/direction_{predict_date}.lgb` 解析。
                                文件不存在时抛 FileNotFoundError，提示「先跑一次训练」。
        predict_date:           `predict_only=True` 时的预测日期（YYYY-MM-DD），用于解析
                                `artifacts/models/direction_{date}.lgb`。None 走今天日期。

    返回 dict：
        训练模式（默认）：
        - predictions / metrics / factor_names / feature_importance / split / model
        - backtest / backtest_metrics / signals（如 run_backtest=True）

        predict_only 模式：
        - predictions:      list[Prediction]，全 (date, ticker) 预测
        - score_frame:      DataFrame[value, score, proba_down, proba_up]
        - model:            DirectionModel（从磁盘加载）
        - predict_model_path: 实际加载的模型路径
        - metrics:          { mode: "predict_only", model_path, n_predictions, total_seconds }
        - factor_names:     模型训练时的因子列名
    """
    t_total = time.time()
    horizon = horizon if horizon is not None else SETTINGS.label.horizon
    threshold = threshold if threshold is not None else SETTINGS.label.direction_threshold
    purge_gap_days = (
        purge_gap_days if purge_gap_days is not None else SETTINGS.split.purge_gap
    )

    # ---------- predict_only 模式短路：1 数据 + 2 因子 + inference ----------
    if predict_only:
        return _run_direction_predict_only(
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
    source = data["source"]  # P7 wiring：让 LLM 因子能按需拉新闻（量价/财务因子忽略此参）
    if verbose:
        logger.info(
            "[1/5] data: prices=%s, moneyflow=%s, financials=%d tickers (%.2fs)",
            tuple(price_panel.shape), tuple(mf_panel.shape) if mf_panel is not None else "—",
            len(financials), time.time() - t0,
        )

    # ---------- 2. 因子 ----------
    t0 = time.time()
    ff = compute_factor_frame(
        price_panel=price_panel,
        moneyflow_panel=mf_panel,
        financials=financials,
        # P7 wiring：把 DataSource.get_news 作为 callable 注入；LLM 因子按 (ticker, date)
        # 区间拉新闻打分。量价/财务/资金流因子的 compute 用 **kwargs 吸收此参，无副作用。
        news_fetcher=source.get_news,
    )
    if verbose:
        logger.info(
            "[2/5] factors: shape=%s, %d factors (%.2fs)",
            tuple(ff.shape), len(ff.factor_names), time.time() - t0,
        )

    # ---------- 3. 标签 + 对齐 ----------
    t0 = time.time()
    y_series = direction_label(
        price_panel, horizon=horizon, threshold=threshold, for_training=True
    )
    X, y = align_xy(
        ff.data, y_series,
        drop_label_nan=True, drop_all_nan_rows=True,
    )
    if verbose:
        logger.info(
            "[3/5] labels+align: X=%s, y=%s, base rate y=1: %.3f (%.2fs)",
            tuple(X.shape), tuple(y.shape), float(y.mean()), time.time() - t0,
        )

    # ---------- 4. 切分（命门）----------
    split: TimeSeriesSplit = time_series_split(
        X.index,
        train_end=train_end,
        valid_end=valid_end,
        purge_gap_days=purge_gap_days,
        label_horizon=horizon,
    )
    if verbose:
        logger.info("[4/5] split: %s", split.summary())

    X_tr, y_tr = X.loc[split.train_mask], y.loc[split.train_mask]
    X_va, y_va = X.loc[split.valid_mask], y.loc[split.valid_mask]
    if X_tr.empty or X_va.empty:
        raise RuntimeError(
            f"切分后 train 或 valid 为空：train={X_tr.shape}, valid={X_va.shape}。"
            "可能 train_end / valid_end 与数据区间不匹配。"
        )

    # ---------- 5. 训练 + 预测 + 评估 ----------
    t0 = time.time()
    model = DirectionModel(**(model_params or {}))
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=early_stopping_rounds,
    )
    score_frame = model.predict_score_frame(X_va)  # 高效路径
    predictions: list[Prediction] = model.predict(X_va)  # 契约形态

    # —— metrics
    y_va_int = y_va.astype(int).values
    scores = score_frame["score"].values
    hard = score_frame["value"].values
    metrics = {
        "train_size": int(split.train_size),
        "valid_size": int(split.valid_size),
        "train_end": split.train_end.date().isoformat(),
        "valid_start": split.valid_start.date().isoformat(),
        "gap_days": int(split.gap_days),
        "label_horizon": int(split.label_horizon),
        "base_rate_train": float(y_tr.mean()),
        "base_rate_valid": float(y_va.mean()),
        "accuracy": float(accuracy_score(y_va_int, hard.astype(int))),
        "auc": _safe_auc(y_va_int, scores),
        "log_loss": _safe_log_loss(y_va_int, scores),
        "n_features": int(X.shape[1]),
        "train_seconds": float(time.time() - t0),
    }
    if verbose:
        logger.info(
            "[5/5] train+eval: AUC=%.4f, Acc=%.4f, LogLoss=%.4f (%.2fs)",
            metrics["auc"], metrics["accuracy"], metrics["log_loss"],
            metrics["train_seconds"],
        )

    if save_model_to:
        model.save(save_model_to)
        # Bug 1 修复：同时落 sidecar metadata.json（含训练 metrics），predict_only 模式加载用
        _save_train_metadata(save_model_to, metrics)
        if verbose:
            logger.info("model + metadata saved → %s", save_model_to)

    # ---------- 6. 回测 + 信号（① 涨跌方向闭环）----------
    backtest_result: BacktestResult | None = None
    signal_report: SignalReport | None = None
    if run_backtest:
        t0 = time.time()
        # 回测区间 = 验证集区间（valid_start ~ valid_end），用验证集的 prediction 喂引擎
        # price_panel 用刚才拉到的「全量」prices（引擎内部用日期 + look-ahead 防线截断）
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

        # 信号生成
        t0 = time.time()
        gen = SignalGenerator(
            buy_threshold=bt_cfg.buy_threshold,
            sell_threshold=bt_cfg.sell_threshold,
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
        # 便利字段（notebook 走读 / 二次分析）
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
# helpers —— 评估指标加点容错
# ===========================================================================

DEFAULT_MODELS_DIR = Path("artifacts/models")
DEFAULT_MODEL_TYPE_DIRECTION = "direction"

# 训练 metrics 中要透传给 predict_only 的字段白名单（renderer 诚信声明显示用）
_TRAIN_METADATA_KEYS_DIRECTION = {
    "train_size", "valid_size", "train_end", "valid_start",
    "gap_days", "label_horizon",
    "base_rate_train", "base_rate_valid",
    "accuracy", "auc", "log_loss",
    "n_features", "train_seconds",
}


def _metadata_path(model_path: Path | str) -> Path:
    """模型旁边的 sidecar metadata.json 路径."""
    p = Path(model_path)
    return p.with_suffix(p.suffix + ".metadata.json")


def _save_train_metadata(model_path: Path | str, metrics: dict[str, Any]) -> None:
    """把训练 metrics 关键字段落到 sidecar JSON（Bug 1 修复）.

    仅保留白名单字段，避免把 sklearn / 内部对象塞进 JSON。
    """
    import json as _json
    keep = {k: v for k, v in metrics.items() if k in _TRAIN_METADATA_KEYS_DIRECTION}
    keep["_saved_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    keep["_model_type"] = DEFAULT_MODEL_TYPE_DIRECTION
    path = _metadata_path(model_path)
    with path.open("w", encoding="utf-8") as f:
        _json.dump(keep, f, ensure_ascii=False, indent=2)


def _load_train_metadata(model_path: Path | str) -> dict[str, Any]:
    """从 sidecar JSON 读训练 metrics；不存在则返回空 dict + 一次性 warning."""
    import json as _json
    path = _metadata_path(model_path)
    if not path.exists():
        logger.warning(
            "未找到训练 metadata sidecar: %s —— 诚信声明指标将显示 N/A。"
            "请用本 pipeline 训练并 save 一次以生成 metadata.", path,
        )
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return _json.load(f)
    except (OSError, _json.JSONDecodeError) as e:
        logger.warning("读取 metadata %s 失败: %s", path, e)
        return {}


def _filter_latest_day(factor_data: pd.DataFrame, universe: list[str] | None) -> pd.DataFrame:
    """从 factor 矩阵里只挑「最新一天 × universe」的行（Bug 2 修复）.

    factor_data 是 MultiIndex=(date, ticker)，包含全历史 panel；
    predict_only 只关心「今天」（即 panel 里最大日期）的 universe 票。
    universe=None → 用最新一天的所有 ticker（不过滤）。
    """
    if factor_data is None or factor_data.empty:
        return factor_data
    # 去掉全 NaN 行
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
    """解析 predict_only 模式下要加载的模型文件路径.

    优先级：
        1. predict_model_path 显式给定 → 直接用
        2. 否则按 `artifacts/models/{model_type}_{predict_date}.lgb` 解析
        3. predict_date 为 None → 用今天日期（系统时区）

    解析后必须文件存在，否则 raise FileNotFoundError 给出明确「先跑训练」提示。
    """
    if predict_model_path is not None:
        path = Path(predict_model_path)
    else:
        date_str = predict_date or _dt.date.today().isoformat()
        path = DEFAULT_MODELS_DIR / f"{model_type}_{date_str}.lgb"
    if not path.exists():
        raise FileNotFoundError(
            f"predict_only 模式找不到模型文件: {path}\n"
            f"请先跑一次训练并 save 到该路径：\n"
            f"  uv run python -c \"from astock_quant.pipeline.run_direction import run_direction; "
            f"run_direction(save_model_to='{path}')\""
        )
    return path


def _run_direction_predict_only(
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

    与训练模式的差异：
    - 跳过 labels（无 y）+ splits + fit + valid metrics
    - 用 align_xy(drop_label_nan=False) 等价做法 —— 这里直接用 ff.data，不需要 y
    - 不跑回测、不算 signals（每日报告由 daily.py 上层组装）
    """
    # 1. 解析模型路径 + 加载
    model_path = _resolve_predict_model_path(
        DEFAULT_MODEL_TYPE_DIRECTION, predict_model_path, predict_date,
    )
    if verbose:
        logger.info("[predict_only] loading model: %s", model_path)
    model = DirectionModel().load(model_path)

    # 2. 数据（若上层已预拉则复用，避免 N 个 pipeline 重复拉同一份数据）
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

    # 3. 因子（compute_factor_frame 不动，与训练路径一致）
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

    # 4. inference —— 只对**最新一天 × universe** 做预测（Bug 2 修复）
    # 训练时模型的 feature_names_ 已保存到 sidecar JSON；model.load 后已恢复。
    # 若 ff.data 列与 feature_names_ 有差异，model.predict 内部会 raise（H1 守门）。
    t0 = time.time()
    X_inf = _filter_latest_day(ff.data, universe)
    score_frame = model.predict_score_frame(X_inf)
    predictions: list[Prediction] = model.predict(X_inf)
    if verbose:
        logger.info(
            "[predict_only 3/3] inference: %d predictions over X=%s (latest day only) (%.2fs)",
            len(predictions), tuple(X_inf.shape), time.time() - t0,
        )

    # 5. 训练 metrics（Bug 1 修复）—— 从 sidecar metadata.json load
    train_metadata = _load_train_metadata(model_path)

    metrics: dict[str, Any] = {
        "mode": "predict_only",
        "model_path": str(model_path),
        "n_predictions": len(predictions),
        "total_seconds": float(time.time() - t_total),
    }
    # 透传训练时的 metric（如 auc / accuracy / log_loss）让 renderer 显示在诚信声明
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


def _safe_auc(y_true, scores) -> float:
    """AUC 在 y 全为同一类时未定义，返回 NaN 而非抛错。"""
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _safe_log_loss(y_true, scores) -> float:
    """log loss clip 分数到 [eps, 1-eps] 防 log(0)。"""
    eps = 1e-15
    scores_clipped = np.clip(scores, eps, 1 - eps)
    return float(log_loss(y_true, scores_clipped, labels=[0, 1]))


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


__all__ = ["run_direction"]
