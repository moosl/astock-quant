"""③ 横截面 Top N 选股 端到端编排 —— P10 实装.

复用 run_direction.py / run_return.py 的 7 步骨架，是 ③ 横截面 Top N 的完整纵向切片。
一个函数串起：

    data.prepare_stage1_data()
        ↓
    factors.compute_factor_frame()     # FactorFrame（与 ① ② 共用，0 改动）
        ↓
    labels.ranking_label()             # 横截面分位数 label（按日期 groupby，守住 look-ahead）
        ↓
    labels.align_xy()                  # 对齐成 (X, y)
        ↓
    models.time_series_split(group_by="date")   # 时序切分 + purge gap + group-aware 校验（命门）
        ↓
    models.RankingModel.fit/predict    # LightGBM 回归分数排序
        ↓
    spearman_corr / NDCG@5 / hit-rate-top5 / top5-bottom5 收益差
        ↓
    BacktestEngine.run(predictions)    # 回测（与 ① ② 共用 engine，引擎已有 Top-K 逻辑）
        ↓
    SignalGenerator.generate(predictions)  # ranking 分支（横截面排序 Top N）

用法：
    from astock_quant.pipeline.run_ranking import run_ranking
    result = run_ranking(top_n=10)
    print(result["metrics"])
    # → {'train_size': ..., 'valid_size': ..., 'spearman_corr': 0.xx, 'ndcg5': 0.xx, ...}
    result["predictions"]   # list[Prediction]，对应验证集每一行
    result["backtest_metrics"]  # 回测指标

"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.stats import spearmanr as _spearmanr
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from astock_quant.backtest.engine import BacktestEngine, BacktestRunConfig
from astock_quant.config.settings import SETTINGS
from astock_quant.contracts import BacktestResult, Prediction, SignalReport
from astock_quant.data.dataset import prepare_stage1_data
from astock_quant.factors.registry import compute_factor_frame
from astock_quant.labels.targets import align_xy, ranking_label
from astock_quant.models.splits import TimeSeriesSplit, time_series_split
from astock_quant.signals.generator import SignalGenerator

from astock_quant.models.ranking import RankingModel

logger = logging.getLogger(__name__)


def run_ranking(
    *,
    universe: list[str] | None = None,
    top_n: int = 10,
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
    """③ 横截面 Top N 完整端到端 —— 训练 + 验证 + 回测 + 信号.

    参数（全部可选；缺省走 SETTINGS）：
        universe:               股票池（同 run_direction H3 修复，已真接通）。
        top_n:                  每日持仓上限（默认 10），透传给回测引擎。
        train_end:              训练截止日，默认 SETTINGS.split.train_end。
        valid_end:              验证截止日，默认 SETTINGS.split.valid_end。
        horizon:                label 未来窗口（交易日），默认 SETTINGS.label.horizon (5)。
        purge_gap_days:         切分时的 purge gap，默认 SETTINGS.split.purge_gap (10)。
        model_params:           LightGBM 超参覆盖（dict），merge 进 RankingModel.DEFAULT_PARAMS。
        early_stopping_rounds:  默认 30；传 None 关闭。
        save_model_to:          产物保存路径（如 "artifacts/ranking_lgbm.txt"）。None 不保存。
        force_refresh_data:     数据缓存强制重拉。默认 False（用缓存）。
        verbose:                True 时打印阶段耗时与中间统计。
        run_backtest:           True 时跑回测（默认 True）。
        backtest_config:        回测配置；None 走默认。
        predict_only:           **P12 新增**：True 时跳过 fit + valid metrics + 回测，从磁盘
                                加载已训练模型，对全 (date, ticker) 做 inference。供每日预测报告用。
        predict_model_path:     `predict_only=True` 时显式指定模型路径。None 则按
                                `artifacts/models/ranking_{predict_date}.lgb` 解析。
        predict_date:           `predict_only=True` 时的预测日期（YYYY-MM-DD），None 走今天。

    返回 dict：
        - predictions:        list[Prediction]，验证集预测（target_type="ranking"）
        - metrics:            { train_size, valid_size, spearman_corr, ndcg5,
                                hit_rate_top5, top5_bottom5_spread, ... }
        - factor_names:       训练用到的因子列名
        - feature_importance: pd.Series（gain 重要性，已排序）
        - split:              TimeSeriesSplit 对象
        - model:              RankingModel（训练好的实例）
        - y_valid:            真实 y_va（横截面 pct rank label）
        - backtest:           BacktestResult（如果 run_backtest=True）
        - backtest_metrics:   回测指标 dict
        - signals:            SignalReport（ranking 分派）
    """
    t_total = time.time()
    horizon = horizon if horizon is not None else SETTINGS.label.horizon
    purge_gap_days = (
        purge_gap_days if purge_gap_days is not None else SETTINGS.split.purge_gap
    )

    # ---------- predict_only 模式短路 ----------
    if predict_only:
        return _run_ranking_predict_only(
            universe=universe,
            top_n=top_n,
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

    # ---------- 2. 因子（与 ① ② 共用 compute_factor_frame，0 改动）----------
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

    # ---------- 3. 标签 + 对齐（ranking_label：横截面分位数，按日期 groupby，守住 look-ahead）---
    t0 = time.time()
    y_series = ranking_label(
        price_panel,
        horizon=horizon,
        for_training=True,
    )
    X, y = align_xy(
        ff.data, y_series,
        drop_label_nan=True, drop_all_nan_rows=True,
    )
    if verbose:
        logger.info(
            "[3/7] labels+align: X=%s, y=%s, y dist: %s (%.2fs)",
            tuple(X.shape), tuple(y.shape),
            {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
            time.time() - t0,
        )

    # ---------- 4. 切分（命门：group_by="date" 守住同 date 不跨集）----------
    split: TimeSeriesSplit = time_series_split(
        X.index,
        train_end=train_end,
        valid_end=valid_end,
        purge_gap_days=purge_gap_days,
        label_horizon=horizon,
        group_by="date",  # P10 命门：横截面 ranking 必须 group-aware
    )
    if verbose:
        logger.info("[4/7] split (group_by=date): %s", split.summary())

    X_tr, y_tr = X.loc[split.train_mask], y.loc[split.train_mask]
    X_va, y_va = X.loc[split.valid_mask], y.loc[split.valid_mask]
    if X_tr.empty or X_va.empty:
        raise RuntimeError(
            f"切分后 train 或 valid 为空：train={X_tr.shape}, valid={X_va.shape}。"
            "可能 train_end / valid_end 与数据区间不匹配。"
        )

    # ---------- 5. 训练 + 预测 + 评估（横截面排序指标）----------
    t0 = time.time()
    model = RankingModel(**(model_params or {}))
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=early_stopping_rounds,
    )
    score_frame = model.predict_score_frame(X_va)
    predictions: list[Prediction] = model.predict(X_va)
    if verbose:
        logger.info("[5/7] RankingModel trained (%.2fs)", time.time() - t0)

    # —— ranking metrics
    y_va_arr = y_va.values.astype(float)
    pred_arr = score_frame["score"].values.astype(float)
    spearman_corr = _calc_spearman(y_va_arr, pred_arr)
    ndcg5 = _calc_ndcg_at_k(y_va, score_frame, k=5)
    hit_rate_top5 = _calc_hit_rate_top5(y_va, score_frame)
    top5_bottom5_spread = _calc_top5_bottom5_spread(y_va, score_frame)

    metrics: dict[str, Any] = {
        "train_size": int(split.train_size),
        "valid_size": int(split.valid_size),
        "train_end": split.train_end.date().isoformat(),
        "valid_start": split.valid_start.date().isoformat(),
        "gap_days": int(split.gap_days),
        "label_horizon": int(split.label_horizon),
        "top_n": top_n,
        "spearman_corr": spearman_corr,
        "ndcg5": ndcg5,
        "hit_rate_top5": hit_rate_top5,
        "top5_bottom5_spread": top5_bottom5_spread,
        "n_features": int(X.shape[1]),
        "train_seconds": float(time.time() - t0),
    }
    if verbose:
        logger.info(
            "[5/7] metrics: spearman=%.4f, NDCG@5=%.4f, hit_top5=%.4f, spread=%.4f (%.2fs)",
            spearman_corr, ndcg5, hit_rate_top5, top5_bottom5_spread,
            metrics["train_seconds"],
        )

    if save_model_to:
        model.save(save_model_to)
        _save_train_metadata(save_model_to, metrics)
        if verbose:
            logger.info("model + metadata saved → %s", save_model_to)

    # ---------- 6. 回测 + 7. 信号 ----------
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
        gen = SignalGenerator()
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
# helpers —— ranking 评估指标
# ===========================================================================

DEFAULT_MODELS_DIR = Path("artifacts/models")
DEFAULT_MODEL_TYPE_RANKING = "ranking"

_TRAIN_METADATA_KEYS_RANKING = {
    "train_size", "valid_size", "train_end", "valid_start",
    "gap_days", "label_horizon", "top_n",
    "spearman_corr", "ndcg5", "hit_rate_top5", "top5_bottom5_spread",
    "n_features", "train_seconds",
}


def _metadata_path(model_path: Path | str) -> Path:
    p = Path(model_path)
    return p.with_suffix(p.suffix + ".metadata.json")


def _save_train_metadata(model_path: Path | str, metrics: dict[str, Any]) -> None:
    """Bug 1 修复：训练 metrics 落 sidecar JSON."""
    import json as _json
    keep = {k: v for k, v in metrics.items() if k in _TRAIN_METADATA_KEYS_RANKING}
    keep["_saved_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    keep["_model_type"] = DEFAULT_MODEL_TYPE_RANKING
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
    """解析 predict_only 模式下要加载的模型文件路径（与 run_direction 同款）."""
    if predict_model_path is not None:
        path = Path(predict_model_path)
    else:
        date_str = predict_date or _dt.date.today().isoformat()
        path = DEFAULT_MODELS_DIR / f"{model_type}_{date_str}.lgb"
    if not path.exists():
        raise FileNotFoundError(
            f"predict_only 模式找不到模型文件: {path}\n"
            f"请先跑一次训练并 save 到该路径：\n"
            f"  uv run python -c \"from astock_quant.pipeline.run_ranking import run_ranking; "
            f"run_ranking(save_model_to='{path}')\""
        )
    return path


def _run_ranking_predict_only(
    *,
    universe: list[str] | None,
    top_n: int,
    force_refresh_data: bool,
    verbose: bool,
    predict_model_path: str | None,
    predict_date: str | None,
    t_total: float,
    prepared_data: dict | None = None,
) -> dict[str, Any]:
    """predict_only 模式：load 模型 + 拉数据 + 算因子 + 全 (date, ticker) predict."""
    model_path = _resolve_predict_model_path(
        DEFAULT_MODEL_TYPE_RANKING, predict_model_path, predict_date,
    )
    if verbose:
        logger.info("[predict_only] loading model: %s", model_path)
    model = RankingModel().load(model_path)

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
        "top_n": top_n,
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


def _calc_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman 相关系数（rank IC）—— 横截面排序的核心指标."""
    if not _SCIPY_OK or len(y_true) < 2:
        return float("nan")
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return float("nan")
    rho, _ = _spearmanr(y_true, y_pred)
    return float(rho) if rho is not None and not np.isnan(rho) else float("nan")


def _calc_ndcg_at_k(y_va, score_frame, *, k: int = 5) -> float:
    """NDCG@K —— 按预测分数排序后，前 K 名的 label 加权收益.

    用验证集内所有日期的平均 NDCG@K。
    """
    try:
        df = score_frame.copy()
        df["label"] = y_va.values
        df.index = y_va.index

        ndcgs = []
        for _date, grp in df.groupby(level="date"):
            if len(grp) < k:
                continue
            grp_sorted = grp.sort_values("score", ascending=False)
            top_k_labels = grp_sorted["label"].iloc[:k].values
            # DCG: 相关度 / log2(rank+1)，这里 label 已是分位数（越大越好）
            dcg = sum(
                float(label) / np.log2(rank + 2)
                for rank, label in enumerate(top_k_labels)
            )
            # IDCG: 理想排序
            ideal_labels = sorted(grp["label"].values, reverse=True)[:k]
            idcg = sum(
                float(label) / np.log2(rank + 2)
                for rank, label in enumerate(ideal_labels)
            )
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        return float(np.mean(ndcgs)) if ndcgs else float("nan")
    except Exception:
        return float("nan")


def _calc_hit_rate_top5(y_va, score_frame) -> float:
    """Hit Rate Top5：预测 Top5 的股票中，真实 label 在前 20% 分位（>= 0.8）的比例.

    ranking_label 是连续 pct rank ∈ [0, 1]，越大越好。
    """
    try:
        df = score_frame.copy()
        df["label"] = y_va.values
        df.index = y_va.index

        hits = []
        for _date, grp in df.groupby(level="date"):
            if len(grp) < 5:
                continue
            top5 = grp.nlargest(5, "score")
            hit = (top5["label"] >= 0.8).sum()
            hits.append(hit / 5.0)
        return float(np.mean(hits)) if hits else float("nan")
    except Exception:
        return float("nan")


def _calc_top5_bottom5_spread(y_va, score_frame) -> float:
    """Top5 - Bottom5 收益差：预测 Top5 的平均 pct-rank label 减 Bottom5 的平均 pct-rank label."""
    try:
        df = score_frame.copy()
        df["label"] = y_va.values
        df.index = y_va.index

        spreads = []
        for _date, grp in df.groupby(level="date"):
            if len(grp) < 10:
                continue
            top5_mean = grp.nlargest(5, "score")["label"].mean()
            bot5_mean = grp.nsmallest(5, "score")["label"].mean()
            spreads.append(top5_mean - bot5_mean)
        return float(np.mean(spreads)) if spreads else float("nan")
    except Exception:
        return float("nan")


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


__all__ = ["run_ranking"]
