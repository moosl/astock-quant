"""信号生成 —— 模型 Prediction → 人可读的买卖 / 持仓信号（SignalReport）.

4 类预测目标共用，按 target_type 分派：
- direction    ① 涨/跌信号（Stage 1 做透）：按预测类别 + 概率给买/持/卖 + 强度
- return       ② 按预期收益率排序给信号
- ranking      ③ Top N 持仓清单
- trade_signal ④ 直接输出买卖点

对应计划 Stage 1 验收的「信号模块能产出可读的买卖/持仓信号」。
Stage 1：把 direction 分支写完整；②③④ 分支留 stub。
"""

from __future__ import annotations

import logging
from typing import Literal

from astock_quant.contracts import (
    Prediction,
    SignalItem,
    SignalReport,
    TargetType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 阈值默认
# ---------------------------------------------------------------------------

# direction（与 BacktestEngine 一致）
DEFAULT_BUY_THRESHOLD = 0.55  # P(涨) ≥ 0.55 → buy
DEFAULT_SELL_THRESHOLD = 0.45  # P(涨) < 0.45 → sell（已持有则清仓）

# return（P9 实装 —— 预测收益率阈值）
# Stage 1 的 stub 用硬编码 ±0.005，P9 升级为可配参数：
# 预测收益率 ≥ +2% → buy；< -2% → sell；中间 → hold
# 阈值挑选理由：模型 horizon=5d、A股 单日波动 1-2%，所以 5 日累计 ±2% 是「值得动手」的最低门槛
DEFAULT_RETURN_BUY_THRESHOLD = 0.02  # +2% 预测收益 → buy
DEFAULT_RETURN_SELL_THRESHOLD = -0.02  # -2% 预测收益 → sell


class SignalGenerator:
    """统一信号生成器 —— 入参 Prediction 列表，按 target_type 分派.

    用法：
        gen = SignalGenerator()
        report = gen.generate(predictions)   # 自动按 target_type 分派
        df = report.to_dataframe()           # 转 DataFrame 写盘 / 喂回测
    """

    def __init__(
        self,
        *,
        buy_threshold: float = DEFAULT_BUY_THRESHOLD,
        sell_threshold: float = DEFAULT_SELL_THRESHOLD,
        ranking_top_n: int = 10,
        return_buy_threshold: float = DEFAULT_RETURN_BUY_THRESHOLD,
        return_sell_threshold: float = DEFAULT_RETURN_SELL_THRESHOLD,
    ) -> None:
        if buy_threshold < sell_threshold:
            raise ValueError(
                f"buy_threshold ({buy_threshold}) 必须 >= sell_threshold ({sell_threshold})"
            )
        if return_buy_threshold < return_sell_threshold:
            raise ValueError(
                f"return_buy_threshold ({return_buy_threshold}) 必须 >= "
                f"return_sell_threshold ({return_sell_threshold})"
            )
        self.buy_threshold = float(buy_threshold)
        self.sell_threshold = float(sell_threshold)
        self.ranking_top_n = int(ranking_top_n)
        self.return_buy_threshold = float(return_buy_threshold)
        self.return_sell_threshold = float(return_sell_threshold)

    def generate(self, predictions: list[Prediction]) -> SignalReport:
        """主入口 —— 按 target_type 分派.

        如果 predictions 跨多种 target_type，按出现频率最高的为主 target_type，
        其余忽略并 warn（Stage 1 不混用）。
        """
        if not predictions:
            return SignalReport(target_type="direction", items=[], notes="empty predictions")

        types = {p.target_type for p in predictions}
        if len(types) > 1:
            # 选最常见的；Stage 1 一次只跑一种 target_type，理论上不会触发
            from collections import Counter
            counter: Counter[TargetType] = Counter(p.target_type for p in predictions)
            primary, _ = counter.most_common(1)[0]
            logger.warning(
                "SignalGenerator.generate: 收到多种 target_type=%s，按主 type=%s 处理，"
                "其余忽略", types, primary,
            )
            predictions = [p for p in predictions if p.target_type == primary]
            target_type: TargetType = primary
        else:
            target_type = next(iter(types))

        if target_type == "direction":
            return self._generate_direction(predictions)
        elif target_type == "return":
            return self._generate_return(predictions)
        elif target_type == "ranking":
            return self._generate_ranking(predictions)
        elif target_type == "trade_signal":
            return self._generate_trade_signal(predictions)
        else:
            raise ValueError(f"未知 target_type: {target_type}")

    # ------------------------------------------------------------------
    # ① direction —— Stage 1 做透
    # ------------------------------------------------------------------

    def _generate_direction(self, predictions: list[Prediction]) -> SignalReport:
        """① 涨跌方向 → buy / hold / sell + 强度.

        规则：
            score >= buy_threshold   → buy
            score <  sell_threshold  → sell
            其余                      → hold

        强度（0~1）：|score - 0.5| × 2。
            - score=0.5 → strength=0（最不确定）
            - score=0.9 或 0.1 → strength=0.8（强信号）

        reason 含模型置信度 + 阈值，方便人读 / 复盘。
        """
        items: list[SignalItem] = []
        for p in predictions:
            score = p.score if p.score is not None else 0.5
            action: Literal["buy", "sell", "hold"]
            if score >= self.buy_threshold:
                action = "buy"
            elif score < self.sell_threshold:
                action = "sell"
            else:
                action = "hold"
            strength = min(1.0, abs(score - 0.5) * 2.0)
            reason = (
                f"模型预测涨概率 {score:.3f}，"
                + (
                    f"≥ 买入阈值 {self.buy_threshold:.2f}" if action == "buy"
                    else f"< 卖出阈值 {self.sell_threshold:.2f}" if action == "sell"
                    else f"位于持有区间 [{self.sell_threshold:.2f}, {self.buy_threshold:.2f})"
                )
            )
            items.append(
                SignalItem(
                    date=p.date,
                    ticker=p.ticker,
                    action=action,
                    strength=float(strength),
                    score=float(score),
                    reason=reason,
                )
            )
        notes = (
            f"target=direction, buy_thr={self.buy_threshold}, sell_thr={self.sell_threshold}, "
            f"n_items={len(items)}, "
            f"n_buy={sum(1 for i in items if i.action == 'buy')}, "
            f"n_sell={sum(1 for i in items if i.action == 'sell')}, "
            f"n_hold={sum(1 for i in items if i.action == 'hold')}"
        )
        return SignalReport(target_type="direction", items=items, notes=notes)

    # ------------------------------------------------------------------
    # ②③④ stub —— 留接入点，签名稳定
    # ------------------------------------------------------------------

    def _generate_return(self, predictions: list[Prediction]) -> SignalReport:
        """② 收益率预测 → 信号（P9 升级：阈值可配 + 强度按阈值缩放）.

        规则（与 direction 同款语义，但阈值是连续 float 而非概率）：
            value ≥ return_buy_threshold   → buy（默认 +2%）
            value <  return_sell_threshold  → sell（默认 -2%）
            其余                              → hold（默认 [-2%, +2%]，模型置信度不足）

        强度（0~1）：
            buy 时   = min(1, (value - buy_thr)  / buy_thr)，超阈越多越强（buy_thr×2 时 = 1）
            sell 时  = min(1, (sell_thr - value) / |sell_thr|)，跌阈越多越强
            hold 时  = 0

        reason 含预测收益率 + 阈值，对应 direction 分支的「P(涨) X，超阈值 Y」格式。
        """
        items: list[SignalItem] = []
        for p in predictions:
            v = float(p.value)
            action: Literal["buy", "sell", "hold"]
            if v >= self.return_buy_threshold:
                action = "buy"
                # 超 buy_threshold 越多越强；阈值的 2 倍处 strength = 1
                strength = min(
                    1.0,
                    (v - self.return_buy_threshold) / max(abs(self.return_buy_threshold), 1e-9),
                )
            elif v < self.return_sell_threshold:
                action = "sell"
                strength = min(
                    1.0,
                    (self.return_sell_threshold - v) / max(abs(self.return_sell_threshold), 1e-9),
                )
            else:
                action = "hold"
                strength = 0.0
            reason = (
                f"模型预测收益率 {v:+.4f}，"
                + (
                    f"≥ 买入阈值 {self.return_buy_threshold:+.3f}" if action == "buy"
                    else f"< 卖出阈值 {self.return_sell_threshold:+.3f}" if action == "sell"
                    else f"位于持有区间 [{self.return_sell_threshold:+.3f}, {self.return_buy_threshold:+.3f})"
                )
            )
            items.append(
                SignalItem(
                    date=p.date, ticker=p.ticker, action=action,
                    strength=float(strength), score=float(v),
                    reason=reason,
                )
            )
        notes = (
            f"target=return, buy_thr={self.return_buy_threshold:+.3f}, "
            f"sell_thr={self.return_sell_threshold:+.3f}, n_items={len(items)}, "
            f"n_buy={sum(1 for i in items if i.action == 'buy')}, "
            f"n_sell={sum(1 for i in items if i.action == 'sell')}, "
            f"n_hold={sum(1 for i in items if i.action == 'hold')}"
        )
        return SignalReport(target_type="return", items=items, notes=notes)

    def _generate_ranking(self, predictions: list[Prediction]) -> SignalReport:
        """③ 横截面排序 → Top N 买入.

        Stage 1 简化：每日按 score 降序取 Top N → buy，其余 → hold。
        """
        items: list[SignalItem] = []
        # 按日分组
        from collections import defaultdict
        by_date: dict = defaultdict(list)
        for p in predictions:
            by_date[p.date].append(p)
        for d, day_preds in by_date.items():
            day_preds.sort(key=lambda x: x.score or 0.0, reverse=True)
            top_set = {p.ticker for p in day_preds[: self.ranking_top_n]}
            for p in day_preds:
                action: Literal["buy", "sell", "hold"] = "buy" if p.ticker in top_set else "hold"
                items.append(
                    SignalItem(
                        date=d, ticker=p.ticker, action=action,
                        strength=float(p.score) if p.score is not None else 0.0,
                        score=float(p.score) if p.score is not None else None,
                        reason=f"日内 rank Top {self.ranking_top_n}" if action == "buy" else "rank 之外",
                    )
                )
        return SignalReport(
            target_type="ranking", items=items,
            notes=f"target=ranking (stub), top_n={self.ranking_top_n}, n_items={len(items)}",
        )

    def _generate_trade_signal(self, predictions: list[Prediction]) -> SignalReport:
        """④ 买卖点 3 分类 → 信号（P11 升级：从 stub 升级为基于 TradeSignalModel 的真实策略）.

        与 TradeSignalModel 契约对齐：
            Prediction.value ∈ {-1.0, 0.0, +1.0}（SL / HOLD / TP）
            Prediction.score = max_proba ∈ [1/3, 1.0]（模型置信度，3 分类下限是 1/3）
            Prediction.proba = None（3 分类完整概率走 model.predict_score_frame）

        映射规则：
            value = +1.0（TP 命中预测）→ action = "buy"
            value = -1.0（SL 命中预测）→ action = "sell"
            value =  0.0（HOLD 预测，horizon 内 TP/SL 都没触达）→ action = "hold"

        强度（0~1）：直接用 score = max_proba。
            - score = 1/3 ≈ 0.333（最低，模型完全不确定，3 类均匀）
            - score = 1.0（完全确定）
            HOLD 类的 strength 也保留 max_proba，便于看「模型对 HOLD 有多确定」。

        reason 含 TP/SL/HOLD 语义 + 置信度，对应 direction 分支的「P(涨) X，超阈值 Y」格式。
        """
        items: list[SignalItem] = []
        for p in predictions:
            v = float(p.value)
            score = float(p.score) if p.score is not None else 1.0 / 3.0
            action: Literal["buy", "sell", "hold"]
            if v > 0.5:  # TP，用 0.5 作为浮点对比，鲁棒于 1.0 / 0.999 等
                action = "buy"
                cls_name = "TP（止盈命中预测）"
            elif v < -0.5:  # SL
                action = "sell"
                cls_name = "SL（止损命中预测）"
            else:  # HOLD（v ≈ 0）
                action = "hold"
                cls_name = "HOLD（horizon 内 TP/SL 都未触达预测）"
            reason = f"模型预测类别 {cls_name}，置信度 {score:.3f}"
            items.append(
                SignalItem(
                    date=p.date, ticker=p.ticker, action=action,
                    strength=score,
                    score=score,
                    reason=reason,
                )
            )
        notes = (
            f"target=trade_signal, n_items={len(items)}, "
            f"n_buy={sum(1 for i in items if i.action == 'buy')}, "
            f"n_sell={sum(1 for i in items if i.action == 'sell')}, "
            f"n_hold={sum(1 for i in items if i.action == 'hold')}"
        )
        return SignalReport(target_type="trade_signal", items=items, notes=notes)


__all__ = [
    "SignalGenerator",
    "DEFAULT_BUY_THRESHOLD",
    "DEFAULT_SELL_THRESHOLD",
    "DEFAULT_RETURN_BUY_THRESHOLD",
    "DEFAULT_RETURN_SELL_THRESHOLD",
]
