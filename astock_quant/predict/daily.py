"""每日预测报告 —— Stage 4 P12.

入口：
    uv run python -m astock_quant.predict.daily --date today
    uv run python -m astock_quant.predict.daily --date 2026-05-16 --output-dir artifacts/daily_reports
    uv run python -m astock_quant.predict.daily --help

依赖：必须先训练 4 个 pipeline 各一次并 save 到 `artifacts/models/{type}_{date}.lgb`。
模型文件不存在时该 target 会报错（其他 3 target 仍尝试执行），所有 target 失败时退出码非 0。

设计：
- 调 4 个 pipeline 的 `predict_only=True` 模式（model-engineer 已实装）
- 每个 target 独立 try/except —— 部分失败不阻塞其他（弱基线学习项目不该一个挂全挂）
- 收集结果组装成 renderer 期望的 schema → 喂 `predict.renderer.render()`
- 完整 results JSON 落 `<output_dir>/predictions_{date}.json` 供 P14 准确率追踪用
- 错误日志落 `<output_dir>/error_{date}.log`
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from astock_quant.config.settings import SETTINGS

logger = logging.getLogger(__name__)

# 4 个 target 的元信息（顺序固定，与 renderer schema 对齐）
_TARGETS = [
    ("direction", "astock_quant.pipeline.run_direction", "run_direction"),
    ("return_", "astock_quant.pipeline.run_return", "run_return"),  # 注意 renderer 用 "return_" 避 keyword
    ("ranking", "astock_quant.pipeline.run_ranking", "run_ranking"),
    ("trade_signal", "astock_quant.pipeline.run_trade_signal", "run_trade_signal"),
]

DEFAULT_OUTPUT_DIR = Path("artifacts/daily_reports")


# ---------------------------------------------------------------------------
# 内部 helpers
# ---------------------------------------------------------------------------


def _resolve_date(date_arg: str | None) -> str:
    """解析 --date 参数 → ISO 日期字符串.

    "today" 或 None → 今天日期；否则按 YYYY-MM-DD 解析（校验格式）。
    """
    if date_arg is None or date_arg.lower() == "today":
        return _dt.date.today().isoformat()
    try:
        return _dt.date.fromisoformat(date_arg).isoformat()
    except ValueError as e:
        raise ValueError(f"--date 必须是 'today' 或 YYYY-MM-DD 格式，实际：{date_arg}") from e


def _call_target(
    module_name: str,
    func_name: str,
    universe: list[str] | None,
    date_str: str,
    prepared_data: dict | None = None,
) -> dict[str, Any]:
    """调单个 pipeline 的 predict_only 模式 —— 失败时抛 exception 给上层捕获.

    prepared_data：上层若已经 prepare_stage1_data 过一次，直接透传复用，
    避免 4 个 pipeline 各自重拉一次 akshare（HS300 4×300=1200 次请求 → 14 小时挂）。
    """
    import importlib

    mod = importlib.import_module(module_name)
    func = getattr(mod, func_name)
    return func(
        universe=universe,
        predict_only=True,
        predict_date=date_str,
        prepared_data=prepared_data,
        verbose=False,
    )


def _serialize_prediction(p: Any) -> dict[str, Any]:
    """Prediction Pydantic → JSON-friendly dict（P14 准确率追踪要读 JSON）."""
    return {
        "ticker": getattr(p, "ticker", None),
        "date": str(getattr(p, "date", "")),
        "target_type": getattr(p, "target_type", None),
        "value": getattr(p, "value", None),
        "score": getattr(p, "score", None),
        "proba": list(getattr(p, "proba", []) or []) or None,
    }


def _strip_non_json(result: dict[str, Any]) -> dict[str, Any]:
    """从 pipeline 返回的 dict 里挑 JSON-friendly 字段；丢掉 model / DataFrame 等."""
    out: dict[str, Any] = {}
    preds = result.get("predictions")
    if preds:
        out["predictions"] = [_serialize_prediction(p) for p in preds]
    buy_preds = result.get("buy_predictions")
    if buy_preds:
        out["buy_predictions"] = [_serialize_prediction(p) for p in buy_preds]
    metrics = result.get("metrics")
    if metrics:
        # 过滤非 JSON 友好类型（如 Path → str）
        out["metrics"] = {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v) for k, v in metrics.items()}
    if "predict_model_path" in result:
        out["predict_model_path"] = str(result["predict_model_path"])
    if "factor_names" in result:
        out["factor_names"] = list(result["factor_names"])
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_daily_predict(
    *,
    date: str | None = None,
    universe: list[str] | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    targets: list[str] | None = None,
    render_report: bool = True,
) -> dict[str, Any]:
    """跑每日预测：调 4 个 pipeline 的 predict_only 模式 → 收集结果 → 渲染报告.

    参数：
        date:           预测日期（YYYY-MM-DD）或 "today" / None。None 走今天。
        universe:       股票池。None 走 SETTINGS.universe。
        output_dir:     报告输出目录。默认 `artifacts/daily_reports`。
        targets:        要跑的 target 子集（"direction"/"return_"/"ranking"/"trade_signal"）。
                        None 跑全部 4 个。
        render_report:  是否调 renderer 出 HTML / Markdown。默认 True。

    返回：dict，含每个 target 的结果 + 错误 + 报告路径。所有 target 失败时
    返回 dict 仍正常返回（不抛），调用者可看 `errors` 字段判断。
    """
    t_total = time.time()
    date_str = _resolve_date(date)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    universe = universe or SETTINGS.universe

    selected = set(targets) if targets else {name for name, _, _ in _TARGETS}

    logger.info("daily predict: date=%s, universe=%d ticker(s), targets=%s",
                date_str, len(universe), sorted(selected))

    # 一次性 prepare 数据（4 个 pipeline 共享，避免 4× 重拉 —— HS300 时这是性命攸关）
    # 失败则 prepared_data=None，下游 pipeline 各自再尝试拉（保留旧行为兜底）
    prepared_data: dict | None = None
    try:
        from astock_quant.data.dataset import prepare_stage1_data
        t_prep = time.time()
        prepared_data = prepare_stage1_data(universe=universe)
        logger.info("prepare_stage1_data: OK (%.2fs, shared across %d targets)",
                    time.time() - t_prep, len(selected))
    except Exception as e:  # noqa: BLE001
        logger.warning("prepare_stage1_data 共享失败，回落到每 pipeline 各自拉：%s", e)

    # 跑 4 个 target，每个独立 try/except
    target_results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    model_paths: list[str] = []

    for name, mod_name, func_name in _TARGETS:
        if name not in selected:
            continue
        try:
            t0 = time.time()
            result = _call_target(mod_name, func_name, universe, date_str,
                                  prepared_data=prepared_data)
            elapsed = time.time() - t0
            logger.info("  %s: OK (%d preds, %.2fs)",
                        name, len(result.get("predictions", [])), elapsed)
            target_results[name] = result
            mp = result.get("predict_model_path")
            if mp:
                model_paths.append(f"{name}={mp}")
        except Exception as e:  # noqa: BLE001 —— 一个 target 挂不该拖垮其他
            msg = f"{name}: {type(e).__name__}: {e}"
            errors.append(msg)
            logger.exception("  %s: FAILED", name)
            target_results[name] = {"error": str(e), "predictions": []}

    # 组装 renderer 期望的 schema
    results: dict[str, Any] = {
        "report_date": date_str,
        "universe_size": len(universe),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "data_cutoff": date_str,
        "total_seconds": float(time.time() - t_total),
        "model_version": date_str,  # 简化：模型版本 = 训练日期
        "model_paths": "; ".join(model_paths) if model_paths else "无",
        "errors": errors,
        "direction": target_results.get("direction", {}),
        "return_": target_results.get("return_", {}),
        "ranking": target_results.get("ranking", {}),
        "trade_signal": target_results.get("trade_signal", {}),
        "accuracy": None,  # P14 待实装
    }

    # JSON 落盘（含完整 metrics + predictions，供 P14 准确率追踪）
    json_path = output_dir / f"predictions_{date_str}.json"
    json_payload = {
        "report_date": date_str,
        "universe_size": len(universe),
        "generated_at": results["generated_at"],
        "total_seconds": results["total_seconds"],
        "errors": errors,
        "model_paths": model_paths,
        "results": {
            name: _strip_non_json(target_results.get(name, {}))
            for name, _, _ in _TARGETS if name in target_results
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)
    results["json_path"] = str(json_path)
    logger.info("predictions JSON → %s", json_path)

    # 错误日志（任何 target 失败时落盘）
    if errors:
        err_path = output_dir / f"error_{date_str}.log"
        with err_path.open("w", encoding="utf-8") as f:
            f.write(f"# daily predict errors — {date_str}\n")
            f.write(f"# generated_at: {results['generated_at']}\n\n")
            for e in errors:
                f.write(f"- {e}\n")
        logger.warning("error log → %s", err_path)

    # 渲染报告
    if render_report:
        try:
            from astock_quant.predict.renderer import render
            html_path, md_path = render(results, output_dir)
            results["html_path"] = str(html_path)
            results["md_path"] = str(md_path)
            logger.info("report → %s + %s", html_path, md_path)
        except Exception as e:  # noqa: BLE001
            msg = f"renderer: {type(e).__name__}: {e}"
            errors.append(msg)
            logger.exception("renderer FAILED")
            results["errors"] = errors

    return results


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="astock_quant.predict.daily",
        description=(
            "A股 每日预测报告生成器 —— 调 4 个 pipeline 的 predict_only 模式，"
            "出 HTML + Markdown + JSON。\n\n"
            "**依赖**：首次运行前必须先训练 4 个 pipeline 各一次，把模型 save 到"
            " artifacts/models/{type}_{date}.lgb。训练命令示例：\n"
            "  uv run python -c \"from astock_quant.pipeline.run_direction import run_direction; "
            "run_direction(save_model_to='artifacts/models/direction_2026-05-16.lgb')\"\n"
            "其他 3 个 pipeline（run_return / run_ranking / run_trade_signal）同款。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--date",
        type=str,
        default="today",
        help="预测日期，'today' 或 YYYY-MM-DD（默认 today）。"
             "也用于解析 artifacts/models/{type}_{date}.lgb",
    )
    p.add_argument(
        "--universe",
        type=str,
        default=None,
        help=(
            "股票池：'stage1'（默认 30 蓝筹）/ 'stage4' 或 'hs300'（沪深 300）/ "
            "逗号分隔 ticker（如 '600519,000858'）。默认走 SETTINGS.universe（30 只蓝筹）"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"报告输出目录（默认 {DEFAULT_OUTPUT_DIR}）",
    )
    p.add_argument(
        "--targets",
        type=str,
        default=None,
        help="要跑的 target 子集，逗号分隔 "
             "(direction/return_/ranking/trade_signal)。默认全跑。",
    )
    p.add_argument(
        "--no-render",
        action="store_true",
        help="跳过 HTML/Markdown 渲染，只落 JSON（debug 用）",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="只输出 WARNING 及以上",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    universe = None
    if args.universe:
        arg = args.universe.strip().lower()
        if arg in ("stage1", "stage4", "hs300", "hs300_universe"):
            from astock_quant.config.settings import get_universe
            stage_name = "stage1" if arg == "stage1" else "stage4"
            try:
                universe = get_universe(stage=stage_name)
                print(f"[INFO] --universe {arg} → {stage_name} ({len(universe)} 只)", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] 加载 {stage_name} universe 失败: {e}", file=sys.stderr)
                return 2
        else:
            universe = [t.strip() for t in args.universe.split(",") if t.strip()]
            if not universe:
                print("[ERROR] --universe 解析为空，请检查逗号分隔的代码或使用 stage1/stage4/hs300", file=sys.stderr)
                return 2

    targets = None
    if args.targets:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
        valid = {name for name, _, _ in _TARGETS}
        invalid = [t for t in targets if t not in valid]
        if invalid:
            print(f"[ERROR] --targets 含未知 target: {invalid}（合法值：{sorted(valid)}）",
                  file=sys.stderr)
            return 2

    try:
        results = run_daily_predict(
            date=args.date,
            universe=universe,
            output_dir=args.output_dir,
            targets=targets,
            render_report=not args.no_render,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] daily predict crashed before main loop: {type(e).__name__}: {e}",
              file=sys.stderr)
        traceback.print_exc()
        return 1

    # 全部 target 失败 → exit 1
    n_targets_attempted = len(targets) if targets else len(_TARGETS)
    n_failed = len(results.get("errors", []))
    if n_failed >= n_targets_attempted:
        print(f"[FAIL] all {n_targets_attempted} targets failed. See error log.",
              file=sys.stderr)
        return 1

    # 部分失败 → exit 0 但 stderr 提示
    if results.get("errors"):
        print(f"[WARN] {n_failed}/{n_targets_attempted} target(s) failed:",
              file=sys.stderr)
        for e in results["errors"]:
            print(f"  - {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
