"""数据缓存 + look-ahead 截断.

逻辑参考 a_stock.py 的 _load_ohlcv_astock，但做了两点强化：
1. 缓存粒度更细 —— 行情按 ticker 存一份「全量历史」CSV，财务/资金流/新闻各自分文件。
2. look-ahead 截断独立成 `truncate_by_date` 工具函数，所有「按 curr_date 取数据」的
   地方都过这道闸 —— 不只是行情。

────────────────────────────────────────────────────────────────────────
为什么 look-ahead bias 是头号大敌
────────────────────────────────────────────────────────────────────────
量化回测里，如果某个交易日 T 的特征里混进了 T 之后才知道的信息（明天的收盘价、
下周才发布的财报…），模型会学到「作弊」的规律，回测漂亮但实盘必崩。
本项目两道防线：
  - 第一道（本文件）：数据进入 panel 前，按 curr_date 截断，date > curr_date 的行一律丢弃。
  - 第二道（models/splits.py）：训练/验证切分时挖 purge gap，防标签的未来窗口泄漏。
本文件负责第一道。`truncate_by_date` 必须被所有时点查询调用 —— 这是纪律。
────────────────────────────────────────────────────────────────────────

缓存策略：
- 缓存即「全量历史」：CSV 里存某 ticker 能拿到的最长历史，不按查询区间切。
  好处是查不同区间不用反复拉网络；截断由 `truncate_by_date` 在读取时做。
- 当日有效性：缓存文件的 mtime 是今天 → 直接用；否则视为过期，需重新拉。
  （A股 日线一天一更，当天内重复跑不必反复打网络。）
- 增量更新由 dataset 层决定「要不要重拉」，cache 只提供 read/write/截断原语。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from astock_quant.config.settings import SETTINGS


# ---------------------------------------------------------------------------
# 缓存路径
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    """缓存根目录，不存在则创建。"""
    d = SETTINGS.data_cache_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(kind: str, ticker: str) -> Path:
    """某类数据某 ticker 的缓存文件路径.

    kind: "prices" / "financials" / "moneyflow" / "news" —— 不同数据分文件存。
    文件名形如 `600519-prices.csv`。
    """
    return _cache_dir() / f"{ticker}-{kind}.csv"


# ---------------------------------------------------------------------------
# 缓存有效性
# ---------------------------------------------------------------------------

def is_fresh(path: Path) -> bool:
    """缓存是否「今天更新过」.

    A股 日线一天一更：当天已拉过的缓存当天内可直接复用。
    跨天后视为过期 —— 由调用方（dataset）决定是否重拉。
    """
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return mtime.date() == datetime.now().date()


# ---------------------------------------------------------------------------
# look-ahead 截断 —— 第一道防线，核心
# ---------------------------------------------------------------------------

def truncate_by_date(
    df: pd.DataFrame,
    curr_date: str | datetime | pd.Timestamp,
    date_col: str = "date",
) -> pd.DataFrame:
    """按 curr_date 截断：只保留 date_col <= curr_date 的行 —— 防未来函数.

    这是本项目防 look-ahead bias 的第一道闸门。任何「站在 curr_date 这个时点
    取历史数据」的调用都必须过这个函数：行情、财务、资金流、新闻无一例外。

    参数：
        df:        待截断的 DataFrame，必须含 date_col 列（或同名 index）。
        curr_date: 当前时点 —— 严格大于它的数据视为「未来」，丢弃。
        date_col:  日期列名，默认 "date"。

    返回：截断后的 DataFrame 副本（升序按 date_col 排序）。空 df 原样返回。
    """
    if df is None or df.empty:
        return df

    cutoff = pd.to_datetime(curr_date)
    out = df.copy()

    # 日期可能在 index 上，也可能在列上 —— 统一拿到一个可比较的 Series
    if date_col in out.columns:
        date_series = pd.to_datetime(out[date_col])
    elif out.index.name == date_col:
        date_series = pd.to_datetime(out.index.to_series())
    else:
        raise KeyError(
            f"truncate_by_date: 找不到日期列 '{date_col}'（既不在 columns 也不是 index 名）"
        )

    out = out[date_series.values <= cutoff]
    # 截断后按日期升序，保证下游拿到的永远是有序历史
    if date_col in out.columns:
        out = out.sort_values(date_col).reset_index(drop=True)
    else:
        out = out.sort_index()
    return out


# ---------------------------------------------------------------------------
# 读写原语
# ---------------------------------------------------------------------------

# A股 ticker 一律是 6 位字符串（如 "000001" / "000858" / "600519"）。
# CSV 里的 ticker 列必须强制走字符串通道 —— 否则 pandas 把全数字的代码
# 自动推断成 int64，前导零丢光（'000858' → 858，'000001' → 1），下游所有
# 「按 ticker 匹配」的逻辑都会对不上。这是 P2 早期遗漏的根因 bug，于补丁中修复。
_TICKER_COL = "ticker"


def _coerce_ticker_column(df: pd.DataFrame) -> pd.DataFrame:
    """把 df 的 ticker 列强制成 6 位字符串（in-place 安全）。

    适用于「读出来」和「即将写入」两端 —— 任何含 ticker 列的 DataFrame 都过这道。
    无 ticker 列则原样返回（不是所有缓存表都带 ticker，但目前 prices / moneyflow 都带）。
    """
    if _TICKER_COL not in df.columns:
        return df
    # astype(str) 防 NaN / int / float 各种来源；zfill(6) 补回丢失的前导零
    df[_TICKER_COL] = df[_TICKER_COL].astype(str).str.zfill(6)
    return df


def write_cache(df: pd.DataFrame, kind: str, ticker: str) -> Path:
    """把某 ticker 某类数据的「全量历史」写入 CSV 缓存.

    约定写入的 df 是该 ticker 能拿到的最长历史（不按查询区间切）。
    写入前会强制 ticker 列为 6 位字符串（双层防御，配合读取侧 dtype 锁定）。
    返回写入的文件路径。
    """
    path = cache_path(kind, ticker)
    df = _coerce_ticker_column(df.copy())
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def read_cache(
    kind: str,
    ticker: str,
    curr_date: str | datetime | pd.Timestamp | None = None,
    date_col: str = "date",
) -> pd.DataFrame | None:
    """读取某 ticker 某类数据的缓存，可选按 curr_date 截断.

    关键：用 `dtype={"ticker": str}` 让 pandas 别把 ticker 当数字推断 ——
    否则 '000858' → 858 丢前导零。读完再 zfill(6) 兜底（防历史缓存里写入时漏掉）。

    参数：
        kind / ticker: 定位缓存文件。
        curr_date:     给定则按它做 look-ahead 截断；None 表示读全量历史
                       （仅限「构建/更新缓存」这类内部用途，不要喂给因子层）。
        date_col:      日期列名。

    返回：DataFrame；缓存不存在返回 None。
    """
    path = cache_path(kind, ticker)
    if not path.exists():
        return None

    # dtype={"ticker": str} 关掉 pandas 的整数推断，保住前导零；
    # 配合 _coerce_ticker_column 兜底（含已经被旧版本写成 int 的历史缓存）
    df = pd.read_csv(
        path,
        on_bad_lines="skip",
        encoding="utf-8",
        dtype={_TICKER_COL: str},
    )
    df = _coerce_ticker_column(df)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

    if curr_date is not None:
        df = truncate_by_date(df, curr_date, date_col=date_col)
    return df
