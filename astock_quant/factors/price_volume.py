"""量价因子 —— 动量 / 均值回归 / 波动率 / 换手率 / 技术指标.

研读参考（理解后用自己的话重写，不 import 任何框架）：
- ai-hedge-fund v1 src/agents/technicals.py：动量 1M/3M/6M、ATR、Bollinger、Z-score
- ai-hedge-fund v2 signals/base.py：RSI 计算的纯 pandas 版本
- TradingAgents-astock dataflows/a_stock.py get_indicators：stockstats 算 MACD/BOLL/RSI/ATR

输入：行情 panel（MultiIndex=(date, ticker)，columns=open/high/low/close/volume/amount），
      由 data/dataset.build_price_panel 产出。

输出：每个因子产出 pd.Series，MultiIndex=(date, ticker)，name=因子名，dtype=float。
      数据不足（如 panel 起始处 rolling 窗口还没填满）的位置一律 NaN —— 由模型层 dropna /
      填充处理。

防 look-ahead：所有 rolling / shift 都只看历史（pandas 默认行为）。panel 进入因子前已被
cache.truncate_by_date 按 curr_date 截断，因子内部再无未来泄漏可能。

────────────────────────────────────────────────────────────────────────
关于 stockstats vs 纯 pandas
────────────────────────────────────────────────────────────────────────
P1 / P2 都提到可以用 stockstats 算 MACD/BOLL/RSI/ATR。实测 stockstats 在 panel
（多 ticker）场景下用起来不方便 —— 它默认对整个 DataFrame 当一只票处理，要按 ticker
逐个 wrap 反而比直接写公式麻烦。所以这里：
- 简单指标（动量、波动率、SMA、RSI、ATR、Bollinger Z-score）—— 纯 pandas 实现，
  groupby ticker 后 rolling，清晰可读。
- MACD 这种「多次 EWM 嵌套」的也直接 pandas 写，比 stockstats 短。
- stockstats 留在依赖里，因为 P1/P2 的 docstring 都写了「可参考它的指标列表」——
  此处仅作为「需要时可加」的备选，不在 Stage 1 主路径上。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from astock_quant.factors.base import BaseFactor


# ===========================================================================
# 通用 helper —— 按 ticker 分组做时序操作（panel 是 MultiIndex=(date, ticker)）
# ===========================================================================

def _by_ticker(panel: pd.DataFrame, col: str) -> "pd.core.groupby.SeriesGroupBy":
    """对 panel 的某列按 ticker 分组（panel level 'ticker' 的 groupby）。"""
    return panel[col].groupby(level="ticker", group_keys=False)


# ===========================================================================
# 1. 动量因子（Momentum）—— 过去 N 日累计收益
# ===========================================================================

class Momentum(BaseFactor):
    """N 日动量：过去 N 个交易日的累计收益率 = close_t / close_{t-N} - 1.

    参考 ai-hedge-fund v1 calculate_momentum_signals 的 1M/3M/6M 思路。Stage 1
    主路径用 5/10/20/60 日，覆盖短中长趋势。
    """

    def __init__(self, window: int) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"momentum_{self.window}d"

    def compute(self, panel: pd.DataFrame, **_) -> pd.Series:
        # pct_change(N) = close.shift(-N) ... 不！pandas pct_change(N) 是 close_t / close_{t-N} - 1
        # 默认看历史，是我们要的
        s = _by_ticker(panel, "close").pct_change(self.window)
        s = self._replace_inf(s)
        s.name = self.name
        return s


# ===========================================================================
# 2. 均值回归（Mean Reversion）—— 价格相对均线的偏离 + RSI
# ===========================================================================

class PriceToMA(BaseFactor):
    """价格 / N 日均线 - 1 —— 衡量当前价格相对均线的偏离.

    正值 = 站在均线上方（多头），负值 = 跌破均线。
    参考 v1 calculate_mean_reversion_signals 的 ma_50 思路。
    """

    def __init__(self, window: int) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"price_to_ma{self.window}"

    def compute(self, panel: pd.DataFrame, **_) -> pd.Series:
        ma = _by_ticker(panel, "close").transform(lambda x: x.rolling(self.window).mean())
        s = panel["close"] / ma - 1.0
        s = self._replace_inf(s)
        s.name = self.name
        return s


class ZScore(BaseFactor):
    """价格相对 N 日均线的 z-score = (close - ma) / std.

    标准化版本的均值回归信号 —— 比 PriceToMA 多除以波动率，
    跨股票/跨期可比性更好。参考 v1 calculate_mean_reversion_signals。
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"zscore_{self.window}"

    def compute(self, panel: pd.DataFrame, **_) -> pd.Series:
        close = panel["close"]
        ma = _by_ticker(panel, "close").transform(lambda x: x.rolling(self.window).mean())
        std = _by_ticker(panel, "close").transform(lambda x: x.rolling(self.window).std())
        s = self._safe_div(close - ma, std)
        s = self._replace_inf(s)
        s.name = self.name
        return s


class RSI(BaseFactor):
    """RSI = 100 - 100 / (1 + 平均涨幅 / 平均跌幅).

    经典 14 日 RSI（动量/超买超卖指标）。算法参考 v1 calculate_rsi 和 v2 _compute_rsi。
    数值在 [0, 100] 区间，>70 超买、<30 超卖（传统阈值）。
    """

    def __init__(self, window: int = 14) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"rsi_{self.window}"

    def compute(self, panel: pd.DataFrame, **_) -> pd.Series:
        def _rsi_one(s: pd.Series) -> pd.Series:
            delta = s.diff()
            gain = delta.where(delta > 0, 0.0).rolling(self.window).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(self.window).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100.0 - 100.0 / (1.0 + rs)

        s = _by_ticker(panel, "close").transform(_rsi_one)
        s = self._replace_inf(s)
        s.name = self.name
        return s


# ===========================================================================
# 3. 波动率因子（Volatility）—— N 日收益率标准差 + ATR
# ===========================================================================

class Volatility(BaseFactor):
    """N 日年化波动率 = 日收益 std * sqrt(252).

    参考 v1 calculate_volatility_signals 的 hist_vol。注意年化系数 252 是
    A股 交易日数（粗略），美股 是 252 也行 —— 跨股票可比性比绝对值更重要。
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"volatility_{self.window}d"

    def compute(self, panel: pd.DataFrame, **_) -> pd.Series:
        ret = _by_ticker(panel, "close").pct_change()
        std = ret.groupby(level="ticker", group_keys=False).transform(
            lambda x: x.rolling(self.window).std()
        )
        s = std * np.sqrt(252)
        s = self._replace_inf(s)
        s.name = self.name
        return s


class ATRPct(BaseFactor):
    """ATR 比例 = ATR(14) / close —— 跨股票可比的波动率代理.

    True Range = max(H-L, |H-PrevC|, |L-PrevC|)。算法参考 v1 calculate_atr。
    除以 close 是为了让茅台（高价）和工行（低价）的波动率口径可比。
    """

    def __init__(self, window: int = 14) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"atr_pct_{self.window}"

    def compute(self, panel: pd.DataFrame, **_) -> pd.Series:
        # True Range 三分量。shift(1) 拿到前一交易日收盘 —— 但跨 ticker shift 会串
        # 数据，所以 prev_close 必须按 ticker 分组 shift。
        high, low, close = panel["high"], panel["low"], panel["close"]
        prev_close = _by_ticker(panel, "close").shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        # 按 ticker 分组 rolling，保持索引顺序
        atr = tr.groupby(level="ticker", group_keys=False).transform(
            lambda x: x.rolling(self.window).mean()
        )
        s = atr / close
        s = self._replace_inf(s)
        s.name = self.name
        return s


# ===========================================================================
# 4. 换手 / 量价因子
# ===========================================================================

class VolumeRatio(BaseFactor):
    """成交量比率 = 当日成交量 / 过去 N 日均成交量.

    > 1 表示放量，< 1 缩量。参考 v1 calculate_momentum_signals 的 volume_momentum。
    A股 上下游策略常用「放量突破」作为信号。
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"volume_ratio_{self.window}d"

    def compute(self, panel: pd.DataFrame, **_) -> pd.Series:
        vol_ma = _by_ticker(panel, "volume").transform(lambda x: x.rolling(self.window).mean())
        s = self._safe_div(panel["volume"], vol_ma)
        s = self._replace_inf(s)
        s.name = self.name
        return s


class AmountMean(BaseFactor):
    """过去 N 日平均成交额（元）—— 流动性代理.

    成交额比成交量更跨股票可比（成交量 * 价格水平差异大）。
    log 化是常见做法，但 Stage 1 先用原始值，留给特征工程处理。
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"amount_mean_{self.window}d"

    def compute(self, panel: pd.DataFrame, **_) -> pd.Series:
        if "amount" not in panel.columns:
            return pd.Series(np.nan, index=panel.index, name=self.name)
        s = _by_ticker(panel, "amount").transform(lambda x: x.rolling(self.window).mean())
        s = self._replace_inf(s)
        s.name = self.name
        return s


# ===========================================================================
# 5. 技术指标 —— MACD / Bollinger 位置
# ===========================================================================

class MACDHist(BaseFactor):
    """MACD Histogram = (EMA12 - EMA26) - signal(EMA9 of MACD).

    经典 MACD 柱状图，参考 a_stock.py get_indicators 的 'macdh'。直接用 pandas
    `ewm(span=...).mean()` 实现，比 wrap 整个 DataFrame 简洁。
    柱状图 > 0 为多头动能，< 0 为空头动能；越大动能越强。
    """

    @property
    def name(self) -> str:
        return "macd_hist"

    def compute(self, panel: pd.DataFrame, **_) -> pd.Series:
        def _macd_one(close: pd.Series) -> pd.Series:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal = macd_line.ewm(span=9, adjust=False).mean()
            return macd_line - signal

        s = _by_ticker(panel, "close").transform(_macd_one)
        s = self._replace_inf(s)
        s.name = self.name
        return s


class BollingerPosition(BaseFactor):
    """Bollinger 位置 = (close - lower) / (upper - lower) ∈ [0, 1].

    0 = 触下轨，1 = 触上轨，0.5 = 在中轨。参考 v1 calculate_mean_reversion_signals
    的 price_vs_bb。upper/lower = MA20 ± 2*std20。
    """

    def __init__(self, window: int = 20, n_std: float = 2.0) -> None:
        self.window = window
        self.n_std = n_std

    @property
    def name(self) -> str:
        return f"boll_pos_{self.window}"

    def compute(self, panel: pd.DataFrame, **_) -> pd.Series:
        close = panel["close"]
        ma = _by_ticker(panel, "close").transform(lambda x: x.rolling(self.window).mean())
        std = _by_ticker(panel, "close").transform(lambda x: x.rolling(self.window).std())
        upper = ma + self.n_std * std
        lower = ma - self.n_std * std
        s = self._safe_div(close - lower, upper - lower)
        s = self._replace_inf(s)
        s.name = self.name
        return s


__all__ = [
    "Momentum",
    "PriceToMA",
    "ZScore",
    "RSI",
    "Volatility",
    "ATRPct",
    "VolumeRatio",
    "AmountMean",
    "MACDHist",
    "BollingerPosition",
]
