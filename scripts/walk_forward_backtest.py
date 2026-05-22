"""scripts/walk_forward_backtest.py — Walk-forward 实盘可行性验证（P26 第二阶段）.

单次切分回测（realistic_backtest.py）显示 4/6 变体「跑赢基准」、超额 20-51%。
但那个回测有两个致命问题：① 验证集已被模型选型反复看过（污染）② 只测了一段
上涨行情。本脚本用 **walk-forward** 戳破这两个问题：

  把时间切成 4 段，每段「只用之前的数据训练、测从没见过的下一段」——
  每一段验证集都是真·样本外，且跨多段不同行情。

用法：
    uv run python scripts/walk_forward_backtest.py

诚实预期：超额大概率从「+50%」大幅缩水，可能到 0 附近甚至转负。这才是真相。
"""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from astock_quant.config.settings import get_universe
from astock_quant.data.dataset import prepare_stage1_data
from astock_quant.pipeline.run_ranking import run_ranking
from astock_quant.pipeline.run_return import run_return

# 复用 realistic_backtest.py 的函数（importlib 加载，不需 scripts 是 package）
_RB = Path(__file__).parent / "realistic_backtest.py"
_spec = importlib.util.spec_from_file_location("realistic_backtest", _RB)
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)
build_topn_predictions = _rb.build_topn_predictions
compute_equal_weight_benchmark = _rb.compute_equal_weight_benchmark
run_one_strategy = _rb.run_one_strategy
REALISTIC_COST = _rb.REALISTIC_COST
OUT_DIR = _rb.OUT_DIR

logger = logging.getLogger(__name__)

# Walk-forward 折（扩展窗口）：每折 train 截止日往后滚，valid 测下一段。
# 数据从 2022-01 起 —— fold1 训练集已有近 3 年。
FOLDS = [
    {"name": "fold1", "train_end": "2024-12-31", "valid_end": "2025-04-30"},
    {"name": "fold2", "train_end": "2025-04-30", "valid_end": "2025-08-31"},
    {"name": "fold3", "train_end": "2025-08-31", "valid_end": "2025-12-31"},
    {"name": "fold4", "train_end": "2025-12-31", "valid_end": "2026-04-30"},
]

# 单次切分回测的超额（来自 realistic_backtest.py report，作对比参照）
_SINGLE_SPLIT_EXCESS = {"ranking": 0.2059, "return": 0.5089}


def _fmt(v, pct: bool = False, nd: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:+.2f}%" if pct else f"{v:.{nd}f}"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("astock_quant.data").setLevel(logging.ERROR)
    t0 = time.time()

    universe = get_universe("stage4")
    print(f"[1/3] 准备数据 + 拿 price_panel（{len(universe)} 只）...", flush=True)
    data = prepare_stage1_data(universe=universe)
    price_panel = data["prices"]

    print(f"[2/3] {len(FOLDS)} 折 × 2 模型 = {len(FOLDS) * 2} 次训练（每折只用之前的数据）...",
          flush=True)
    fold_preds: dict = {"ranking": [], "return": []}
    for fold in FOLDS:
        for model, fn in [("ranking", run_ranking), ("return", run_return)]:
            r = fn(
                universe=universe,
                train_end=fold["train_end"], valid_end=fold["valid_end"],
                run_backtest=False, save_model_to=None, verbose=False,
            )
            preds = r["predictions"]
            fold_preds[model].extend(preds)
            print(f"    {fold['name']} · {model}: {len(preds)} 条样本外 predictions",
                  flush=True)

    # 拼接后的 predictions 日期范围
    all_dates = sorted({p.date for preds in fold_preds.values() for p in preds})
    bt_start, bt_end = all_dates[0], all_dates[-1]
    date_level = price_panel.index.get_level_values(0)
    mask = (date_level >= pd.Timestamp(bt_start)) & (date_level <= pd.Timestamp(bt_end))
    price_panel_bt = price_panel[mask]
    benchmark = compute_equal_weight_benchmark(price_panel_bt)

    print("[3/3] 对拼接后的「连续样本外」predictions 跑回测 ...", flush=True)
    results: list[dict] = []
    for model in ("ranking", "return"):
        variant = {
            "name": f"{model}·Top5·周频·walk-forward",
            "model": model, "top_n": 5, "rebal": 5, "cost": REALISTIC_COST,
        }
        try:
            res = run_one_strategy(variant, fold_preds[model], price_panel_bt, benchmark)
        except Exception as e:  # noqa: BLE001
            res = {"variant": variant["name"], "model": model, "error": repr(e)[:200]}
        res["single_split_excess"] = _SINGLE_SPLIT_EXCESS.get(model)
        results.append(res)
        print(f"    {model}: walk-forward 超额={_fmt(res.get('excess_return_annualized'), pct=True)}"
              f"（单次切分曾是 {_fmt(_SINGLE_SPLIT_EXCESS.get(model), pct=True)}）", flush=True)

    # 报告
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    n_months = round((pd.Timestamp(bt_end) - pd.Timestamp(bt_start)).days / 30, 1)
    report = _assemble_report(results, bt_start, bt_end, n_months)

    (OUT_DIR / f"walk_forward_{date_str}.json").write_text(
        json.dumps({"folds": FOLDS, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT_DIR / f"walk_forward_report_{date_str}.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\n总耗时 {time.time() - t0:.0f}s")


def _assemble_report(results: list[dict], bt_start, bt_end, n_months) -> str:
    lines = [
        f"# Walk-forward 实盘可行性验证报告 — {datetime.now():%Y-%m-%d}",
        "",
        "> 这是 P26 第二阶段。单次切分回测显示「4/6 变体跑赢、超额 20-51%」，",
        "> 但验证集被污染、只测一段涨市。walk-forward 让每段验证集都是真·样本外。",
        "",
        "## 验证设置",
        "",
        f"- {len(FOLDS)} 折扩展窗口，每折「只用之前数据训练、测下一段」",
        f"- 回测区间：{bt_start} ~ {bt_end}（约 {n_months} 个月，跨 4 段不同行情）",
        "- 真实成本往返 ~0.42%，策略 Top5·周频调仓，等权基准对比",
        "",
        "## 折设置",
        "",
        "| 折 | 训练截止 | 测试段（真·样本外）|",
        "|---|---|---|",
    ]
    for f in FOLDS:
        lines.append(f"| {f['name']} | ≤ {f['train_end']} | {f['train_end']} ~ {f['valid_end']} |")

    lines += [
        "",
        "## 结果：walk-forward vs 单次切分",
        "",
        "| 策略 | 单次切分超额(年化) | **walk-forward 超额(年化)** | 总收益 | 夏普 | 最大回撤 | 信息比率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['variant']} | — | ERROR | {r['error'][:40]} | | | |")
            continue
        lines.append(
            f"| {r['variant']} | {_fmt(r.get('single_split_excess'), pct=True)} | "
            f"**{_fmt(r.get('excess_return_annualized'), pct=True)}** | "
            f"{_fmt(r.get('total_return'), pct=True)} | {_fmt(r.get('sharpe'))} | "
            f"{_fmt(r.get('max_drawdown'), pct=True)} | {_fmt(r.get('information_ratio'))} |"
        )

    # 判定
    valid = [r for r in results if "error" not in r and r.get("excess_return_annualized") is not None]
    passed = [r for r in valid
              if r["excess_return_annualized"] > 0 and (r.get("information_ratio") or -1) > 0.5]
    lines += ["", "## 判定", ""]
    if not valid:
        lines.append("⚠️ 所有变体回测出错，无法判定。")
    elif passed:
        lines.append(f"**{len(passed)}/{len(valid)} 个策略在 walk-forward 下仍跑赢基准。**")
        lines.append("")
        lines.append("⚠️ 即便如此，仍有幸存者偏差、财务 look-ahead、样本量小、成交假设乐观等"
                     "**无法补救**的偏差。walk-forward 解决了「验证集污染 + 单一行情」，"
                     "但没解决其余偏差 —— 通过 walk-forward 是「必要条件」，不是「充分条件」。")
    else:
        lines.append(f"**0/{len(valid)} 个策略在 walk-forward 下跑赢基准。**")
        lines.append("")
        lines.append("**诚实结论：单次切分回测的「超额 20-51%」是验证集污染 + 单一涨市制造的"
                     "假象。** 一旦换成真·样本外的滚动验证，超额消失。这套量化**不具备实盘价值**，"
                     "请勿用真钱跟单。建议继续作为学习项目。")

    lines += [
        "", "---", "",
        "## ⚠️ 仍然存在、walk-forward 也修不掉的偏差",
        "",
        "即使本报告结果不好看（大概率），也别以为「修一修就能实盘」。下列偏差 walk-forward "
        "解决不了：",
        "",
        "1. **幸存者偏差** —— 仍用「今天的」沪深300成分回测历史，股票池是赢家池。",
        "2. **财务 look-ahead** —— 财务因子按报告期末日而非实际发布日截断。",
        "3. **成交假设乐观** —— 收盘价成交、固定滑点、无冲击成本。",
        "4. **信号本身弱** —— 模型 IC≈0.02 / macro-F1≈0.41，弱信号样本外还会再衰减。",
        "",
        "这些偏差**全部让回测好于实盘**。所以：walk-forward 跑输 = 基本可断定不能实盘；"
        "walk-forward 跑赢 = 也只是「还没被证伪」，离「能实盘」仍很远。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
