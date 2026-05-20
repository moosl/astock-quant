"""命令行入口 —— 薄壳，解析参数后调 astock_quant.pipeline.

用法：
    # 默认：跑 ① 涨跌方向 完整链路（30 只蓝筹起步池）
    uv run python scripts/run_pipeline.py

    # 自定义 universe / 训练截止日 / 验证截止日
    uv run python scripts/run_pipeline.py --universe 600519,000858 --train-end 2025-06-30 --valid-end 2026-05-01

    # 只跑训练，跳过回测（更快）
    uv run python scripts/run_pipeline.py --no-backtest

    # 调回测阈值
    uv run python scripts/run_pipeline.py --buy-threshold 0.51 --sell-threshold 0.49 --max-positions 5

Stage 1 只接 ① 涨跌方向（target=direction）；②③④ 留待后续阶段。
"""

from __future__ import annotations

import argparse
import logging
import sys

from astock_quant.backtest.engine import BacktestRunConfig
from astock_quant.pipeline.run_direction import run_direction


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="A股 量化 ① 涨跌方向 端到端 pipeline（数据→因子→标签→训练→回测→信号）",
    )
    p.add_argument(
        "--target",
        choices=["direction"],
        default="direction",
        help="预测目标（Stage 1 只支持 direction）",
    )
    p.add_argument(
        "--universe",
        type=str,
        default=None,
        help="股票池，逗号分隔的 6 位代码（如 '600519,000858'），缺省走 SETTINGS.universe",
    )
    p.add_argument("--train-end", type=str, default=None, help="训练集截止日 (YYYY-MM-DD)")
    p.add_argument("--valid-end", type=str, default=None, help="验证集截止日 (YYYY-MM-DD)")
    p.add_argument("--horizon", type=int, default=None, help="label 未来窗口（交易日）")
    p.add_argument(
        "--force-refresh-data",
        action="store_true",
        help="忽略缓存全部重拉",
    )
    p.add_argument("--no-backtest", action="store_true", help="跳过回测（更快，只评估模型）")
    p.add_argument("--buy-threshold", type=float, default=None, help="回测买入阈值")
    p.add_argument("--sell-threshold", type=float, default=None, help="回测卖出阈值")
    p.add_argument("--max-positions", type=int, default=None, help="同时持仓上限")
    p.add_argument(
        "--missing-prediction-action",
        choices=["liquidate", "hold"],
        default=None,
        help="持仓票当日无 prediction 时的策略（liquidate=清仓 / hold=维持）",
    )
    p.add_argument("--save-model-to", type=str, default=None, help="模型产物保存路径")
    p.add_argument("--quiet", action="store_true", help="只输出关键 metrics")
    return p.parse_args(argv)


def _fmt_metric(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_summary(result: dict) -> None:
    """打印关键 metrics —— 给命令行用户一个清爽的摘要."""
    print()
    print("=" * 60)
    print("训练 / 评估指标")
    print("=" * 60)
    m = result.get("metrics", {})
    for k in [
        "train_size", "valid_size", "train_end", "valid_start",
        "base_rate_train", "base_rate_valid",
        "accuracy", "auc", "log_loss",
        "n_features", "train_seconds", "total_seconds",
    ]:
        if k in m:
            print(f"  {k:24s} = {_fmt_metric(m[k])}")

    if "backtest_metrics" in result:
        print()
        print("=" * 60)
        print("回测指标")
        print("=" * 60)
        bm = result["backtest_metrics"]
        for k in [
            "start_date", "end_date", "trading_days",
            "total_return", "annualized_return",
            "sharpe", "sortino", "max_drawdown", "max_drawdown_date",
            "n_trades", "n_buy_orders", "n_sell_orders",
            "win_rate", "profit_loss_ratio",
            "n_rejected_constraint", "rejection_reasons",
        ]:
            if k in bm:
                print(f"  {k:24s} = {_fmt_metric(bm[k])}")

    if "signals" in result:
        print()
        print("信号摘要:", result["signals"].notes)
    print()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.target != "direction":
        # Stage 1 只支持 direction，argparse choices 已经守住，这里防御编程
        print(f"❌ Stage 1 只支持 target=direction，不支持 {args.target}", file=sys.stderr)
        return 2

    universe = None
    if args.universe:
        universe = [t.strip() for t in args.universe.split(",") if t.strip()]
        if not universe:
            print("❌ --universe 解析为空，请检查逗号分隔的代码", file=sys.stderr)
            return 2

    # 回测 config（仅在传了相关参数 / 不跳过回测时构造）
    bt_cfg: BacktestRunConfig | None = None
    if not args.no_backtest and any(
        v is not None for v in (
            args.buy_threshold, args.sell_threshold,
            args.max_positions, args.missing_prediction_action,
        )
    ):
        bt_cfg = BacktestRunConfig()
        if args.buy_threshold is not None:
            bt_cfg.buy_threshold = args.buy_threshold
        if args.sell_threshold is not None:
            bt_cfg.sell_threshold = args.sell_threshold
        if args.max_positions is not None:
            bt_cfg.max_positions = args.max_positions
        if args.missing_prediction_action is not None:
            bt_cfg.missing_prediction_action = args.missing_prediction_action

    result = run_direction(
        universe=universe,
        train_end=args.train_end,
        valid_end=args.valid_end,
        horizon=args.horizon,
        save_model_to=args.save_model_to,
        force_refresh_data=args.force_refresh_data,
        verbose=not args.quiet,
        run_backtest=not args.no_backtest,
        backtest_config=bt_cfg,
    )

    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
