"""时序安全的数据切分 —— 防 look-ahead bias 的第二道防线.

（第一道防线是 data/cache.py 按 curr_date 截断行情数据。）

────────────────────────────────────────────────────────────────────────
为什么要 purge gap：命门
────────────────────────────────────────────────────────────────────────
direction label 的语义是「T 日的标签 = T+horizon 日的真实涨跌」（见 labels/targets.py）。
所以训练集里 date=T 的样本，其 y 实际「看了」T+horizon 日的真实价格。

如果训练 / 验证按 train_end 这一刀简单切：
    训练集：date ∈ [start, train_end]
    验证集：date ∈ [train_end + 1trading_day, valid_end]
那么训练集尾部 date=train_end-k（k < horizon）的样本，其标签的"未来窗口"会跨过
切线落进验证集 —— 模型在训练时已"看过"验证集前 horizon 天的真实价格答案。这就是
经典的 label leakage / look-ahead bias。

防线：在切线后挖掉一个 gap：
    训练集：date ∈ [start, train_end]
    gap（丢弃）：(train_end, train_end + gap_days]      ← 含 horizon 个交易日的"未来"
    验证集：date ∈ (train_end + gap_days, valid_end]

数学约束：**gap_days >= label_horizon**。否则训练集尾部样本的未来窗口仍会
泄漏到验证集。本模块在切分时严格校验，传 gap_days < horizon 直接抛 ValueError。

实现细节：
- gap 按「实际出现在 index 中的交易日数」计，不按自然日。这样周末 / 假期不会让
  gap 在自然日维度看着够、交易日维度不够。
- 用 panel 的 date level 去重排序后按位置切，比"加 N 个自然日 + 重新筛"鲁棒。

切分参数从 config/settings.py 的 SplitConfig 读，但 split 函数都允许显式传 ——
方便 walk-forward 等场景灵活配置。

────────────────────────────────────────────────────────────────────────
group_by="date" 模式（横截面 ranking 场景专用）
────────────────────────────────────────────────────────────────────────
time_series_split 的默认行为是按日期连续切（date 维度），同一 date 的所有 ticker
天然在同一集，对 direction/return 任务是安全的。

但为了守住这个不变量（防止将来有人改 splits 逻辑时破坏它），time_series_split
支持 group_by="date" 参数：显式按 date 整体分组切，并在切后校验「同 date 不跨集」。

适用场景：
  - None / 不传（默认）：time series 二分类 / 回归，按日期连续切，不做 group 校验
  - "date"：横截面 ranking 场景，显式要求同 date 的所有 (date, ticker) 行必须全在
    训练集 OR 全在验证集，切后严格校验，违反则抛 ValueError
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from astock_quant.config.settings import SETTINGS


# ===========================================================================
# 输出契约
# ===========================================================================

@dataclass
class TimeSeriesSplit:
    """时序切分结果.

    属性：
        train_mask:    pd.Series[bool]，与输入 index 等长，True 表示该样本归训练集
        valid_mask:    pd.Series[bool]，同上，True 表示归验证集
        train_end:     训练集最后一个交易日（pd.Timestamp）
        valid_start:   验证集第一个交易日（pd.Timestamp），>= train_end + gap_days
        gap_days:      实际使用的 purge gap 交易日数
        label_horizon: 用于校验的 label 未来窗口（与 gap_days 配套）
        purged_count:  被 gap 丢弃的样本数（落在 (train_end, valid_start) 区间的）
    """

    train_mask: pd.Series
    valid_mask: pd.Series
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    gap_days: int
    label_horizon: int
    purged_count: int

    @property
    def train_size(self) -> int:
        return int(self.train_mask.sum())

    @property
    def valid_size(self) -> int:
        return int(self.valid_mask.sum())

    def summary(self) -> str:
        return (
            f"TimeSeriesSplit(train={self.train_size}, valid={self.valid_size}, "
            f"purged={self.purged_count}, gap_days={self.gap_days}, "
            f"horizon={self.label_horizon}, train_end={self.train_end.date()}, "
            f"valid_start={self.valid_start.date()})"
        )


# ===========================================================================
# 切分主函数
# ===========================================================================

def time_series_split(
    index: pd.MultiIndex | pd.DatetimeIndex,
    *,
    train_end: str | pd.Timestamp | None = None,
    valid_end: str | pd.Timestamp | None = None,
    purge_gap_days: int | None = None,
    label_horizon: int | None = None,
    group_by: str | None = None,
) -> TimeSeriesSplit:
    """按时间切分训练 / 验证集，带 purge gap.

    参数：
        index:           样本索引。可以是 MultiIndex=(date, ticker) 或纯 DatetimeIndex。
                         切分按 date level 做（如果是 MultiIndex 从中取 date level）。
        train_end:       训练集结束日（含），默认 SETTINGS.split.train_end
        valid_end:       验证集结束日（含），默认 SETTINGS.split.valid_end
        purge_gap_days:  训练 / 验证间挖掉的「交易日数」（按 index 实际出现的日期算），
                         默认 SETTINGS.split.purge_gap
        label_horizon:   label 的未来窗口（用于校验 gap），默认 SETTINGS.label.horizon
        group_by:        None（默认）或 "date"。
                         - None：time series 二分类/回归场景，按日期连续切，不做额外校验。
                         - "date"：横截面 ranking 场景，切后严格校验「同一 date 的所有
                           (date, ticker) 行必须全在训练集 OR 全在验证集」，违反则抛
                           ValueError。此模式要求 index 为 MultiIndex(date, ticker)。

    返回：
        TimeSeriesSplit 对象。`train_mask` / `valid_mask` 都是 pd.Series（index=输入 index，
        dtype=bool），可直接 `X.loc[split.train_mask]` 取训练集。

    抛错：
        ValueError: 当 `purge_gap_days < label_horizon` —— 这是命门，会让标签泄漏。
        ValueError: 当 train_end >= valid_end，或两者超出 index 的日期范围。
        ValueError: 当 group_by="date" 且切分后存在同 date 跨集的情况。
    """
    train_end = pd.Timestamp(train_end or SETTINGS.split.train_end)
    valid_end = pd.Timestamp(valid_end or SETTINGS.split.valid_end)
    gap = int(purge_gap_days if purge_gap_days is not None else SETTINGS.split.purge_gap)
    horizon = int(label_horizon if label_horizon is not None else SETTINGS.label.horizon)

    # ——— 命门校验 ———
    if gap < horizon:
        raise ValueError(
            f"purge_gap_days={gap} 小于 label_horizon={horizon} —— 标签会从训练集"
            f"泄漏到验证集（look-ahead bias）。请把 gap 调到 >= {horizon}。"
        )
    if train_end >= valid_end:
        raise ValueError(f"train_end={train_end.date()} 必须早于 valid_end={valid_end.date()}")

    # 从 index 拿 date 序列
    dates = _date_series_from_index(index)

    # 把 train_end / valid_end 对齐到 index 中实际出现的交易日（向前对齐）
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    if unique_dates.empty:
        raise ValueError("index 没有任何日期数据，无法切分")

    train_end_aligned = _align_to_trading_day(unique_dates, train_end, direction="back")
    valid_end_aligned = _align_to_trading_day(unique_dates, valid_end, direction="back")

    if train_end_aligned is None or train_end_aligned < unique_dates[0]:
        raise ValueError(
            f"train_end={train_end.date()} 早于数据起始 {unique_dates[0].date()}"
        )
    if valid_end_aligned is None or valid_end_aligned <= train_end_aligned:
        raise ValueError(
            f"valid_end={valid_end.date()} 不晚于（对齐后的）train_end={train_end_aligned.date()}"
        )

    # 计算 valid_start：从 train_end_aligned 在 unique_dates 中的位置往后跳 gap+1 个交易日
    train_end_pos = unique_dates.get_loc(train_end_aligned)
    valid_start_pos = train_end_pos + gap + 1
    if valid_start_pos >= len(unique_dates):
        raise ValueError(
            f"训练集尾 {train_end_aligned.date()} + gap {gap} 个交易日超出数据范围；"
            f"unique trading days 仅 {len(unique_dates)} 个"
        )
    valid_start = unique_dates[valid_start_pos]
    if valid_start > valid_end_aligned:
        raise ValueError(
            f"valid_start={valid_start.date()} 已晚于 valid_end={valid_end_aligned.date()} "
            "—— 验证集为空，请扩大 valid_end 或缩小 gap"
        )

    # 构造 masks
    train_mask = (dates >= unique_dates[0]) & (dates <= train_end_aligned)
    valid_mask = (dates >= valid_start) & (dates <= valid_end_aligned)
    purged_mask = (dates > train_end_aligned) & (dates < valid_start)

    result = TimeSeriesSplit(
        train_mask=train_mask,
        valid_mask=valid_mask,
        train_end=train_end_aligned,
        valid_start=valid_start,
        gap_days=gap,
        label_horizon=horizon,
        purged_count=int(purged_mask.sum()),
    )

    if group_by == "date":
        _validate_group_aware(index, result)
    elif group_by is not None:
        raise ValueError(f"group_by 只支持 None 或 'date'，收到 {group_by!r}")

    return result


# ===========================================================================
# helpers
# ===========================================================================

def _date_series_from_index(index) -> pd.Series:
    """从 index 拿 date 序列，返回 pd.Series[Timestamp] index=输入 index, dtype=datetime64.

    支持：
        - MultiIndex 含 name='date' 的 level
        - MultiIndex 第 0 level 是日期
        - DatetimeIndex
        - 普通 Index of datetime-likes
    """
    if isinstance(index, pd.MultiIndex):
        try:
            level_vals = index.get_level_values("date")
        except KeyError:
            level_vals = index.get_level_values(0)
        s = pd.Series(pd.to_datetime(level_vals), index=index)
    elif isinstance(index, pd.DatetimeIndex):
        s = pd.Series(index, index=index)
    else:
        s = pd.Series(pd.to_datetime(index), index=index)
    return s


def _align_to_trading_day(
    trading_days: pd.DatetimeIndex,
    target: pd.Timestamp,
    *,
    direction: str = "back",
) -> pd.Timestamp | None:
    """把 target 对齐到 trading_days 中实际存在的最近交易日.

    direction="back"：返回 <= target 的最大交易日（不存在则 None）。
    direction="forward"：返回 >= target 的最小交易日。
    """
    arr = trading_days.values.astype("datetime64[ns]")
    t = np.datetime64(pd.Timestamp(target), "ns")
    if direction == "back":
        mask = arr <= t
        if not mask.any():
            return None
        return pd.Timestamp(arr[mask].max())
    elif direction == "forward":
        mask = arr >= t
        if not mask.any():
            return None
        return pd.Timestamp(arr[mask].min())
    else:
        raise ValueError(f"unknown direction: {direction}")


def _validate_group_aware(index: pd.MultiIndex | pd.DatetimeIndex, split: TimeSeriesSplit) -> None:
    """group_by='date' 命门校验：同一 date 的所有行必须全在训练集或全在验证集.

    横截面 ranking 场景下，同一日的 ticker 共享同一日市场环境，ranking_label 是相对值。
    若同日不同 ticker 被切到不同集，等于「看着今天涨没涨来预测今天哪些票会涨」——
    这是横截面维度的 look-ahead bias。

    此函数在 time_series_split 末尾被调用，校验失败直接抛 ValueError。
    """
    if not isinstance(index, pd.MultiIndex):
        raise ValueError("group_by='date' 要求 index 为 MultiIndex(date, ticker)")

    dates = _date_series_from_index(index)
    train_dates = set(dates[split.train_mask].unique())
    valid_dates = set(dates[split.valid_mask].unique())
    overlap_dates = train_dates & valid_dates
    if overlap_dates:
        sample_dates = sorted(overlap_dates)[:3]
        raise ValueError(
            f"group_by='date' 校验失败：以下 date 同时出现在训练集和验证集中，"
            f"违反横截面 look-ahead 防线：{[d.date() for d in sample_dates]}..."
        )


__all__ = ["TimeSeriesSplit", "time_series_split"]
