"""每日预测准确率追踪 —— Stage 4 P14.

回看 daily.py 落盘的 `artifacts/daily_reports/predictions_YYYY-MM-DD.json` 历史预测，
拉真实 T+horizon 收盘价做 ground truth，算每类预测的命中率。

入口：
    uv run python -m astock_quant.predict.accuracy --days 30
    uv run python -m astock_quant.predict.accuracy --days 30 --target direction
    uv run python -m astock_quant.predict.accuracy --days 30 --ticker 600519
    uv run python -m astock_quant.predict.accuracy --days 30 --output-dir artifacts/daily_reports

设计：
- **不重新调模型** —— 只读历史 JSON + 拉真实行情算指标
- **horizon 截断** —— 还没到期的预测（T+horizon > 今天）一律跳过
- **缺失友好** —— JSON 缺失 / ticker 真实价格拿不到 → 跳过 + warning，不中断
- **诚信结论** —— 命中率接近 50%（猜硬币）/Sharpe < 1 → 模型仍是弱基线，与 ① AUC=0.5131 一脉相承

4 类指标（设计 §4.2）：
- direction：T 信号 buy/sell → T+horizon 真实涨跌方向一致 → hit
- return：value > 0 ↔ actual_return > 0 同向 → hit + MAE
- ranking：每日 score 降序 Top-K，真实涨幅前 K/2 的比例（precision@K/2）+ spearman
- trade_signal：value=+1 → 真实路径先触 TP（+5%） / value=-1 → 先触 SL（-3%） / value=0 → 都没触 → hit
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from astock_quant.config.settings import SETTINGS

logger = logging.getLogger(__name__)

DEFAULT_REPORTS_DIR = Path("artifacts/daily_reports")
DEFAULT_HORIZON = 5  # 与 SETTINGS.label.horizon 对齐（不读 SETTINGS 直接，避免循环）
DEFAULT_TP_PCT = 0.05  # trade_signal_label 默认
DEFAULT_SL_PCT = -0.03


# ---------------------------------------------------------------------------
# 历史 JSON 扫描
# ---------------------------------------------------------------------------


def _list_prediction_files(
    reports_dir: Path,
    start_date: _dt.date,
    end_date: _dt.date,
) -> list[Path]:
    """扫 `predictions_YYYY-MM-DD.json`，过滤到 [start_date, end_date] 区间."""
    files: list[Path] = []
    for p in sorted(reports_dir.glob("predictions_*.json")):
        # 文件名格式：predictions_2026-05-16.json
        stem = p.stem  # predictions_2026-05-16
        date_part = stem.removeprefix("predictions_")
        try:
            d = _dt.date.fromisoformat(date_part)
        except ValueError:
            logger.warning("跳过无法解析日期的文件：%s", p)
            continue
        if start_date <= d <= end_date:
            files.append(p)
    return files


def _load_predictions(json_path: Path) -> dict[str, Any] | None:
    """读单个 prediction JSON 文件，失败 → 返回 None + warning."""
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("读取 %s 失败：%s", json_path, e)
        return None


# ---------------------------------------------------------------------------
# Ground truth 拉取（带缓存）
# ---------------------------------------------------------------------------


class _GroundTruthCache:
    """单只 ticker 的 ground truth 缓存：拉一次 close 序列重复用.

    避免每个 prediction 都打一次网络。一只 ticker 在一次 evaluate_predictions
    调用中只 fetch 一次（覆盖区间 [min_date - 30 buffer, max_date + horizon + 10 buffer]）。
    """

    def __init__(self, source, start_buffer_days: int = 30, end_buffer_days: int = 15) -> None:
        self._source = source
        self._cache: dict[str, pd.Series] = {}  # ticker → close 序列（DatetimeIndex）
        self._missing: set[str] = set()  # 拉过空数据的 ticker，跳过重试
        self._start_buf = start_buffer_days
        self._end_buf = end_buffer_days
        self._global_start: _dt.date | None = None
        self._global_end: _dt.date | None = None

    def set_date_range(self, start: _dt.date, end: _dt.date) -> None:
        """设定全局日期范围（call 在 fetch 之前）."""
        self._global_start = start - _dt.timedelta(days=self._start_buf)
        self._global_end = end + _dt.timedelta(days=self._end_buf)

    def get_close_series(self, ticker: str) -> pd.Series | None:
        """拿单只 ticker 的 close 序列（DatetimeIndex），失败返回 None."""
        if ticker in self._cache:
            return self._cache[ticker]
        if ticker in self._missing:
            return None
        if self._global_start is None or self._global_end is None:
            raise RuntimeError("先调 set_date_range")
        bars = self._source.get_prices(
            ticker,
            self._global_start.isoformat(),
            self._global_end.isoformat(),
        )
        if not bars:
            logger.warning("ground truth 拉取失败 / 空：%s", ticker)
            self._missing.add(ticker)
            return None
        # bars 是 list[PriceBar]
        idx = pd.DatetimeIndex([pd.Timestamp(b.date) for b in bars])
        s = pd.Series([b.close for b in bars], index=idx, name=ticker).sort_index()
        self._cache[ticker] = s
        return s


def _future_close_at_horizon(
    close_series: pd.Series,
    t_date: _dt.date,
    horizon: int,
) -> tuple[float | None, float | None]:
    """从 close 序列拿 (close[T], close[T+horizon])，按交易日数前推 horizon 个 bar.

    返回 (close_t, close_t_plus_h)；任一不可得 → (None, None)。
    """
    t_ts = pd.Timestamp(t_date)
    # 找 T 当日（或之后第一个交易日）的位置
    pos = close_series.index.searchsorted(t_ts, side="left")
    if pos >= len(close_series):
        return None, None
    if close_series.index[pos] != t_ts:
        # T 当日没数据（停牌 / 非交易日）→ 找最近一个 T 之前的交易日作为 entry
        if pos == 0:
            return None, None
        pos -= 1
    entry = float(close_series.iloc[pos])
    target_pos = pos + horizon
    if target_pos >= len(close_series):
        return None, None  # 还没到期
    future = float(close_series.iloc[target_pos])
    return entry, future


def _future_close_path(
    close_series: pd.Series,
    t_date: _dt.date,
    horizon: int,
) -> tuple[float | None, list[float] | None]:
    """拿 close[T] 和 close[T+1..T+horizon] 路径（trade_signal 用）."""
    t_ts = pd.Timestamp(t_date)
    pos = close_series.index.searchsorted(t_ts, side="left")
    if pos >= len(close_series):
        return None, None
    if close_series.index[pos] != t_ts:
        if pos == 0:
            return None, None
        pos -= 1
    entry = float(close_series.iloc[pos])
    path_end = pos + horizon + 1
    if path_end > len(close_series):
        return None, None  # 路径不完整
    path = [float(x) for x in close_series.iloc[pos + 1: path_end].values]
    return entry, path


# ---------------------------------------------------------------------------
# 4 类 evaluator
# ---------------------------------------------------------------------------


def _eval_direction(
    predictions: list[dict[str, Any]],
    t_date: _dt.date,
    gt_cache: _GroundTruthCache,
    horizon: int,
    ticker_filter: str | None,
    stats: dict[str, Any],
) -> None:
    """① 涨跌方向命中率：T 预测 buy/sell → T+h 真实涨/跌方向一致.

    value=1.0 → 预测涨（buy）；value=0.0 → 预测跌（sell）。
    """
    for p in predictions:
        ticker = p.get("ticker")
        if ticker_filter and ticker != ticker_filter:
            continue
        value = p.get("value")
        if ticker is None or value is None:
            continue
        s = gt_cache.get_close_series(ticker)
        if s is None:
            stats["n_missing_gt"] += 1
            continue
        entry, future = _future_close_at_horizon(s, t_date, horizon)
        if entry is None or future is None or entry <= 0:
            stats["n_horizon_unreached"] += 1
            continue
        actual_return = future / entry - 1.0
        predicted_up = value >= 0.5
        actual_up = actual_return > 0
        hit = predicted_up == actual_up
        stats["n_evaluated"] += 1
        if hit:
            stats["n_hit"] += 1
        stats["sum_actual_return"] += actual_return
        if predicted_up:
            stats["n_buy"] += 1
            stats["sum_buy_actual_return"] += actual_return
        else:
            stats["n_sell"] += 1


def _eval_return(
    predictions: list[dict[str, Any]],
    t_date: _dt.date,
    gt_cache: _GroundTruthCache,
    horizon: int,
    ticker_filter: str | None,
    stats: dict[str, Any],
) -> None:
    """② 收益率回归：预测 vs 真实收益率，算 MAE + 方向一致率."""
    for p in predictions:
        ticker = p.get("ticker")
        if ticker_filter and ticker != ticker_filter:
            continue
        pred_return = p.get("value")
        if ticker is None or pred_return is None:
            continue
        s = gt_cache.get_close_series(ticker)
        if s is None:
            stats["n_missing_gt"] += 1
            continue
        entry, future = _future_close_at_horizon(s, t_date, horizon)
        if entry is None or future is None or entry <= 0:
            stats["n_horizon_unreached"] += 1
            continue
        actual_return = future / entry - 1.0
        err = abs(pred_return - actual_return)
        stats["n_evaluated"] += 1
        stats["sum_abs_err"] += err
        stats["sum_actual_return"] += actual_return
        # 方向一致
        if (pred_return > 0) == (actual_return > 0):
            stats["n_dir_correct"] += 1


def _eval_ranking(
    predictions: list[dict[str, Any]],
    t_date: _dt.date,
    gt_cache: _GroundTruthCache,
    horizon: int,
    ticker_filter: str | None,
    stats: dict[str, Any],
    top_k: int = 5,
) -> None:
    """③ 横截面排名：每日按 score 降序取 Top K，看真实涨幅前 K/2 比例 + spearman."""
    if not predictions:
        return
    # ticker_filter 在 ranking 任务下意义不大（横截面用全 universe），但保留兼容
    if ticker_filter:
        predictions = [p for p in predictions if p.get("ticker") == ticker_filter]
        if not predictions:
            return

    # 同一日所有预测一起算横截面 rank
    rows = []
    for p in predictions:
        ticker = p.get("ticker")
        score = p.get("score")
        if ticker is None or score is None:
            continue
        s = gt_cache.get_close_series(ticker)
        if s is None:
            continue
        entry, future = _future_close_at_horizon(s, t_date, horizon)
        if entry is None or future is None or entry <= 0:
            continue
        rows.append({"ticker": ticker, "score": float(score), "actual": future / entry - 1.0})

    if len(rows) < 2:
        stats["n_horizon_unreached"] += len(predictions) - len(rows)
        return

    df = pd.DataFrame(rows)
    stats["n_evaluated"] += len(df)

    # Top K precision@K/2：score 降序前 K，看其中真实涨幅前 K/2 的比例
    k = min(top_k, len(df))
    top_by_score = df.nlargest(k, "score")
    top_by_actual = set(df.nlargest(max(k // 2, 1), "actual")["ticker"])
    hits = sum(1 for t in top_by_score["ticker"] if t in top_by_actual)
    stats["sum_topk_precision"] += hits / max(k // 2, 1)
    stats["n_days"] += 1

    # Spearman 等价于 rank 一致性
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(df["score"], df["actual"])
        if rho is not None and not pd.isna(rho):
            stats["sum_spearman"] += float(rho)
            stats["n_spearman_days"] += 1
    except ImportError:
        pass


def _eval_trade_signal(
    predictions: list[dict[str, Any]],
    t_date: _dt.date,
    gt_cache: _GroundTruthCache,
    horizon: int,
    ticker_filter: str | None,
    stats: dict[str, Any],
    tp_pct: float = DEFAULT_TP_PCT,
    sl_pct: float = DEFAULT_SL_PCT,
) -> None:
    """④ 三元 buy/sell/hold：真实路径先触 TP/SL/都没触 → 与预测对齐 → 命中.

    用真实 close 路径模拟 trade_signal_label 的判定规则。
    """
    for p in predictions:
        ticker = p.get("ticker")
        if ticker_filter and ticker != ticker_filter:
            continue
        value = p.get("value")
        if ticker is None or value is None:
            continue
        s = gt_cache.get_close_series(ticker)
        if s is None:
            stats["n_missing_gt"] += 1
            continue
        entry, path = _future_close_path(s, t_date, horizon)
        if entry is None or path is None or entry <= 0:
            stats["n_horizon_unreached"] += 1
            continue
        # 真实标签：模拟 trade_signal_label 路径判定
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)
        actual_label = 0.0  # 默认 HOLD
        for px in path:
            if px >= tp_price:
                actual_label = 1.0
                break
            if px <= sl_price:
                actual_label = -1.0
                break

        stats["n_evaluated"] += 1
        if int(value) == int(actual_label):
            stats["n_hit"] += 1

        # buy 信号（value=+1）盈利与否
        if int(value) == 1:
            stats["n_buy"] += 1
            if actual_label == 1.0:
                stats["n_buy_profit"] += 1


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def evaluate_predictions(
    *,
    start_date: str | _dt.date,
    end_date: str | _dt.date,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    source: Any = None,
    horizon: int = DEFAULT_HORIZON,
    target: str | None = None,
    ticker: str | None = None,
    tp_pct: float = DEFAULT_TP_PCT,
    sl_pct: float = DEFAULT_SL_PCT,
) -> dict[str, Any]:
    """评估 [start_date, end_date] 区间所有日内 predictions 的命中率.

    参数：
        start_date / end_date:  评估区间 (date 或 ISO 字符串)
        reports_dir:            predictions_*.json 所在目录
        source:                 DataSource（默认 AStockSource()）
        horizon:                T+N 横向 (默认 5)
        target:                 只评估指定 target（"direction" / "return_" / "ranking" / "trade_signal"）
        ticker:                 只评估指定 ticker
        tp_pct / sl_pct:        trade_signal 用的 TP/SL 阈值

    返回 dict：
        {
          "start_date": "...", "end_date": "...", "horizon": 5,
          "n_files_scanned": N, "n_files_used": M,
          "today_cutoff": "YYYY-MM-DD",   # 横向不达期的截止
          "results": {
            "direction": {hit_rate, n_evaluated, n_hit, n_missing_gt, ...},
            "return_": {mae, dir_correct_rate, ...},
            "ranking": {avg_topk_precision, avg_spearman, ...},
            "trade_signal": {hit_rate, buy_profit_rate, ...},
          }
        }
    """
    if isinstance(start_date, str):
        start_date = _dt.date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = _dt.date.fromisoformat(end_date)
    reports_dir = Path(reports_dir)

    if source is None:
        from astock_quant.data.astock_source import AStockSource

        source = AStockSource()

    # 截止：T+horizon 必须 ≤ 今天，否则路径不完整
    today = _dt.date.today()
    today_cutoff = today - _dt.timedelta(days=horizon)

    files = _list_prediction_files(reports_dir, start_date, end_date)
    logger.info("扫描 %d 个 predictions JSON 文件（区间 %s ~ %s）",
                len(files), start_date, end_date)

    # 初始化 4 类 stats（None 表示该 target 没遇到数据）
    stats: dict[str, dict[str, Any]] = {
        "direction": defaultdict(float),
        "return_": defaultdict(float),
        "ranking": defaultdict(float),
        "trade_signal": defaultdict(float),
    }

    gt_cache = _GroundTruthCache(source)
    gt_cache.set_date_range(start_date, end_date)

    n_used = 0
    n_skipped_horizon = 0

    selected_targets = {target} if target else {"direction", "return_", "ranking", "trade_signal"}

    for f in files:
        # 文件名日期 = 预测日 T
        t_date = _dt.date.fromisoformat(f.stem.removeprefix("predictions_"))
        if t_date > today_cutoff:
            n_skipped_horizon += 1
            logger.info("跳过 %s：T+%d (=%s) > 今天 (%s)，路径不完整",
                        t_date, horizon, t_date + _dt.timedelta(days=horizon), today)
            continue
        payload = _load_predictions(f)
        if payload is None:
            continue
        n_used += 1

        results_section = payload.get("results", {})
        for tgt_name, evaluator in [
            ("direction", _eval_direction),
            ("return_", _eval_return),
            ("ranking", _eval_ranking),
            ("trade_signal", _eval_trade_signal),
        ]:
            if tgt_name not in selected_targets:
                continue
            tgt_data = results_section.get(tgt_name) or {}
            preds = tgt_data.get("predictions") or []
            if not preds:
                continue
            if tgt_name == "trade_signal":
                evaluator(preds, t_date, gt_cache, horizon, ticker, stats[tgt_name],
                          tp_pct=tp_pct, sl_pct=sl_pct)
            else:
                evaluator(preds, t_date, gt_cache, horizon, ticker, stats[tgt_name])

    # 汇总指标
    summary_results: dict[str, Any] = {}

    # direction
    d = stats["direction"]
    if d["n_evaluated"] > 0:
        summary_results["direction"] = {
            "n_evaluated": int(d["n_evaluated"]),
            "n_hit": int(d["n_hit"]),
            "hit_rate": d["n_hit"] / d["n_evaluated"],
            "n_buy": int(d["n_buy"]),
            "n_sell": int(d["n_sell"]),
            "avg_actual_return": d["sum_actual_return"] / d["n_evaluated"],
            "avg_buy_actual_return": (
                d["sum_buy_actual_return"] / d["n_buy"] if d["n_buy"] > 0 else None
            ),
            "n_missing_gt": int(d["n_missing_gt"]),
            "n_horizon_unreached": int(d["n_horizon_unreached"]),
        }

    # return
    r = stats["return_"]
    if r["n_evaluated"] > 0:
        summary_results["return_"] = {
            "n_evaluated": int(r["n_evaluated"]),
            "mae": r["sum_abs_err"] / r["n_evaluated"],
            "dir_correct_rate": r["n_dir_correct"] / r["n_evaluated"],
            "avg_actual_return": r["sum_actual_return"] / r["n_evaluated"],
            "n_missing_gt": int(r["n_missing_gt"]),
            "n_horizon_unreached": int(r["n_horizon_unreached"]),
        }

    # ranking
    rk = stats["ranking"]
    if rk["n_evaluated"] > 0:
        summary_results["ranking"] = {
            "n_evaluated": int(rk["n_evaluated"]),
            "n_days": int(rk["n_days"]),
            "avg_topk_precision": (
                rk["sum_topk_precision"] / rk["n_days"] if rk["n_days"] > 0 else None
            ),
            "avg_spearman": (
                rk["sum_spearman"] / rk["n_spearman_days"]
                if rk["n_spearman_days"] > 0 else None
            ),
            "n_spearman_days": int(rk["n_spearman_days"]),
        }

    # trade_signal
    ts = stats["trade_signal"]
    if ts["n_evaluated"] > 0:
        summary_results["trade_signal"] = {
            "n_evaluated": int(ts["n_evaluated"]),
            "n_hit": int(ts["n_hit"]),
            "hit_rate": ts["n_hit"] / ts["n_evaluated"],
            "n_buy": int(ts["n_buy"]),
            "n_buy_profit": int(ts["n_buy_profit"]),
            "buy_profit_rate": (
                ts["n_buy_profit"] / ts["n_buy"] if ts["n_buy"] > 0 else None
            ),
            "n_missing_gt": int(ts["n_missing_gt"]),
            "n_horizon_unreached": int(ts["n_horizon_unreached"]),
        }

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "horizon": horizon,
        "today_cutoff": today_cutoff.isoformat(),
        "n_files_scanned": len(files),
        "n_files_used": n_used,
        "n_skipped_horizon": n_skipped_horizon,
        "ticker_filter": ticker,
        "target_filter": target,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "results": summary_results,
    }


# ---------------------------------------------------------------------------
# 报告渲染（Markdown + HTML 简版）
# ---------------------------------------------------------------------------


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.2f}%"


def _fmt_num(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def _format_honesty_verdict(results: dict[str, Any]) -> str:
    """诚信结论：四类指标接近基线时明示弱基线（与 P3b/P9 一脉相承）.

    direction hit_rate ~50% → 跟猜硬币没差
    return MAE 与 std 同量级 → 没 alpha
    ranking spearman ~0 → 没排序能力
    trade_signal hit_rate ~1/3 → 接近三类均匀基线
    """
    lines: list[str] = []
    r = results.get("results", {})

    dir_ = r.get("direction")
    if dir_:
        hr = dir_["hit_rate"]
        delta_bp = (hr - 0.5) * 10000
        verdict = "接近 50%（跟猜硬币没差）" if abs(hr - 0.5) < 0.02 else ("略高于猜硬币" if hr > 0.5 else "略低于猜硬币")
        lines.append(
            f"- ① 涨跌方向命中率 {hr * 100:.2f}%（{verdict}），相对 50% 基线 {delta_bp:+.0f} bp"
        )

    rt = r.get("return_")
    if rt:
        dc = rt["dir_correct_rate"]
        delta_bp = (dc - 0.5) * 10000
        lines.append(
            f"- ② 收益率方向一致率 {dc * 100:.2f}%（相对 50% {delta_bp:+.0f} bp），MAE={_fmt_num(rt['mae'])}"
        )

    rk = r.get("ranking")
    if rk:
        sp = rk.get("avg_spearman")
        verdict = "接近 0（没排序能力）" if sp is not None and abs(sp) < 0.05 else "略有信号" if sp and abs(sp) >= 0.05 else "—"
        lines.append(
            f"- ③ 横截面排名 spearman={_fmt_num(sp)}（{verdict}），Top-K precision={_fmt_num(rk.get('avg_topk_precision'))}"
        )

    ts = r.get("trade_signal")
    if ts:
        hr = ts["hit_rate"]
        bp = ts.get("buy_profit_rate")
        baseline = 1.0 / 3  # 三类均匀基线
        delta_bp = (hr - baseline) * 10000
        lines.append(
            f"- ④ 买卖信号 3 类命中率 {hr * 100:.2f}%（相对 33.3% 基线 {delta_bp:+.0f} bp）"
            f"，buy 信号真实命中 TP 率 {_fmt_pct(bp)}"
        )

    if not lines:
        return "_无可评估的预测_"

    overall = (
        "\n\n**整体判断**：上述指标若持续接近随机基线（hit_rate ~50%、spearman ~0、"
        "3 类 hit ~33%），与 Stage 1 ① direction AUC=0.5131 / P9 ② return R²=-0.002 一脉相承，"
        "**模型仍是诚信弱基线**，没有 alpha。学习/研究项目的预期结果，**不要当真盘**。"
    )
    return "\n".join(lines) + overall


def render_accuracy_report(
    results: dict[str, Any],
    output_dir: Path | str,
    as_of: _dt.date | None = None,
) -> tuple[Path, Path]:
    """渲染 Markdown + HTML 准确率报告."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = as_of or _dt.date.today()
    md_path = output_dir / f"accuracy_{as_of.isoformat()}.md"
    html_path = output_dir / f"accuracy_{as_of.isoformat()}.html"

    # Markdown
    r = results.get("results", {})
    md_lines = [
        f"# 准确率追踪报告 — {as_of.isoformat()}",
        "",
        f"- 评估区间：{results['start_date']} ~ {results['end_date']}",
        f"- horizon：{results['horizon']} 个交易日",
        f"- 横向截止：{results['today_cutoff']}（之后的预测路径不完整，跳过）",
        f"- 扫描 JSON：{results['n_files_scanned']} 个，使用 {results['n_files_used']} 个，",
        f"  因路径不完整跳过 {results['n_skipped_horizon']} 个",
    ]
    if results.get("ticker_filter"):
        md_lines.append(f"- ticker 过滤：{results['ticker_filter']}")
    if results.get("target_filter"):
        md_lines.append(f"- target 过滤：{results['target_filter']}")
    md_lines.extend(["", "## 各类指标", ""])

    if "direction" in r:
        d = r["direction"]
        md_lines.extend([
            "### ① 涨跌方向 (direction)",
            f"- 样本数：{d['n_evaluated']}（命中 {d['n_hit']}）",
            f"- **命中率：{d['hit_rate'] * 100:.2f}%**",
            f"- buy={d['n_buy']}, sell={d['n_sell']}",
            f"- 验证集真实平均收益率：{_fmt_pct(d['avg_actual_return'])}",
            f"- buy 信号真实平均收益率：{_fmt_pct(d['avg_buy_actual_return'])}",
            f"- 缺 ground truth：{d['n_missing_gt']}，未到期：{d['n_horizon_unreached']}",
            "",
        ])
    if "return_" in r:
        rt = r["return_"]
        md_lines.extend([
            "### ② 收益率回归 (return)",
            f"- 样本数：{rt['n_evaluated']}",
            f"- **MAE：{_fmt_num(rt['mae'])}**",
            f"- 方向一致率：{rt['dir_correct_rate'] * 100:.2f}%",
            f"- 真实平均收益率：{_fmt_pct(rt['avg_actual_return'])}",
            f"- 缺 ground truth：{rt['n_missing_gt']}，未到期：{rt['n_horizon_unreached']}",
            "",
        ])
    if "ranking" in r:
        rk = r["ranking"]
        md_lines.extend([
            "### ③ 横截面排名 (ranking)",
            f"- 样本数：{rk['n_evaluated']}（覆盖 {rk['n_days']} 天）",
            f"- **Top-K precision@K/2 平均：{_fmt_num(rk.get('avg_topk_precision'))}**",
            f"- **Spearman 平均：{_fmt_num(rk.get('avg_spearman'))}**（{rk['n_spearman_days']} 天有效）",
            "",
        ])
    if "trade_signal" in r:
        ts = r["trade_signal"]
        md_lines.extend([
            "### ④ 买卖信号 (trade_signal)",
            f"- 样本数：{ts['n_evaluated']}（命中 {ts['n_hit']}）",
            f"- **3 类命中率：{ts['hit_rate'] * 100:.2f}%**（基线 1/3 = 33.33%）",
            f"- buy 信号 {ts['n_buy']} 个，真实触 TP {ts['n_buy_profit']} 个，"
            f"buy 命中率 {_fmt_pct(ts.get('buy_profit_rate'))}",
            f"- 缺 ground truth：{ts['n_missing_gt']}，未到期：{ts['n_horizon_unreached']}",
            "",
        ])

    md_lines.extend(["## 诚信结论", "", _format_honesty_verdict(results), ""])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # HTML（简版，复用 Markdown 内容）
    html = (
        f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>准确率追踪报告 — {as_of.isoformat()}</title>"
        f"<style>body{{font-family:-apple-system,sans-serif;max-width:880px;margin:40px auto;padding:0 20px;color:#222}}"
        f"h1{{border-bottom:2px solid #2c5282;padding-bottom:6px}}"
        f"h3{{color:#2c5282;margin-top:24px}}"
        f"pre{{background:#f7fafc;padding:12px;border-radius:6px;overflow-x:auto}}"
        f"</style></head><body>"
        f"<pre>{_html_escape(md_path.read_text(encoding='utf-8'))}</pre>"
        f"</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")

    return md_path, html_path


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="astock_quant.predict.accuracy",
        description=(
            "A股 每日预测准确率追踪 —— 扫历史 predictions JSON，拉真实价格算命中率。\n\n"
            "**依赖**：先用 daily.py 跑过几天 + 距今 ≥ horizon（默认 5）个交易日，"
            "否则预测路径不完整无法评估。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--days", type=int, default=30, help="回看天数（默认 30）")
    p.add_argument("--start-date", type=str, default=None,
                   help="评估起始日（YYYY-MM-DD），覆盖 --days")
    p.add_argument("--end-date", type=str, default=None,
                   help="评估终止日（YYYY-MM-DD），默认今天")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                   help=f"预测视野（交易日，默认 {DEFAULT_HORIZON}）")
    p.add_argument("--reports-dir", type=str, default=str(DEFAULT_REPORTS_DIR),
                   help=f"predictions_*.json 所在目录（默认 {DEFAULT_REPORTS_DIR}）")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_REPORTS_DIR),
                   help=f"准确率报告输出目录（默认 {DEFAULT_REPORTS_DIR}）")
    p.add_argument("--target", type=str, default=None,
                   choices=["direction", "return_", "ranking", "trade_signal"],
                   help="只评估指定 target（默认全部）")
    p.add_argument("--ticker", type=str, default=None, help="只评估指定 ticker")
    p.add_argument("--tp-pct", type=float, default=DEFAULT_TP_PCT,
                   help=f"trade_signal TP 阈值（默认 {DEFAULT_TP_PCT}）")
    p.add_argument("--sl-pct", type=float, default=DEFAULT_SL_PCT,
                   help=f"trade_signal SL 阈值（默认 {DEFAULT_SL_PCT}）")
    p.add_argument("--no-render", action="store_true",
                   help="不渲染 Markdown/HTML 报告，只打印")
    p.add_argument("--quiet", action="store_true", help="只输出 WARNING+")
    # 兼容性占位：让 SETTINGS import 不要抛
    _ = SETTINGS  # noqa: F841
    return p


def _print_summary(results: dict[str, Any]) -> None:
    """命令行简表."""
    print()
    print("=" * 60)
    print(f"准确率追踪：{results['start_date']} ~ {results['end_date']} (horizon={results['horizon']})")
    print(f"  扫描 {results['n_files_scanned']} 个 JSON，使用 {results['n_files_used']} 个，"
          f"跳过 {results['n_skipped_horizon']} 个未到期")
    print("=" * 60)
    r = results.get("results", {})
    if "direction" in r:
        d = r["direction"]
        print(f"① direction: hit_rate={d['hit_rate'] * 100:.2f}% "
              f"(n={d['n_evaluated']}, hit={d['n_hit']}, buy={d['n_buy']}, sell={d['n_sell']})")
    if "return_" in r:
        rt = r["return_"]
        print(f"② return:    MAE={_fmt_num(rt['mae'])}, "
              f"dir_correct={rt['dir_correct_rate'] * 100:.2f}% (n={rt['n_evaluated']})")
    if "ranking" in r:
        rk = r["ranking"]
        print(f"③ ranking:   spearman={_fmt_num(rk.get('avg_spearman'))}, "
              f"top-K prec={_fmt_num(rk.get('avg_topk_precision'))} ({rk['n_days']} 天)")
    if "trade_signal" in r:
        ts = r["trade_signal"]
        print(f"④ trade:     hit_rate={ts['hit_rate'] * 100:.2f}% "
              f"(n={ts['n_evaluated']}, buy={ts['n_buy']}, buy_profit={ts['n_buy_profit']})")
    if not r:
        print("(无可评估的预测 —— 检查 reports_dir 是否有 predictions JSON / 日期是否到期)")
    print()
    print(_format_honesty_verdict(results))
    print()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # 解析日期范围
    end_date = (
        _dt.date.fromisoformat(args.end_date) if args.end_date else _dt.date.today()
    )
    if args.start_date:
        start_date = _dt.date.fromisoformat(args.start_date)
    else:
        start_date = end_date - _dt.timedelta(days=args.days)

    try:
        results = evaluate_predictions(
            start_date=start_date,
            end_date=end_date,
            reports_dir=args.reports_dir,
            horizon=args.horizon,
            target=args.target,
            ticker=args.ticker,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] evaluate_predictions crashed: {type(e).__name__}: {e}",
              file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    _print_summary(results)

    if not args.no_render:
        try:
            md_path, html_path = render_accuracy_report(results, args.output_dir)
            logger.info("Markdown → %s", md_path)
            logger.info("HTML → %s", html_path)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] render failed: {type(e).__name__}: {e}", file=sys.stderr)

    if not results.get("results"):
        # 没评估到任何预测 → exit 1 提示用户先跑 daily.py
        print("[WARN] 没有可评估的预测。请先用 daily.py 跑几天预测，"
              f"且距今 ≥ {args.horizon} 个交易日。", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
