"""LLM 情绪 / 事件因子 —— Stage 2 P6 实现.

═══════════════════════════════════════════════════════════════════════════
  Stage 2 P6 已实现 LLMNewsSentiment（之前是 Stage 1 stub）.
═══════════════════════════════════════════════════════════════════════════

LLM 在本系统的角色：把新闻 / 研报 / 公告转成情绪 / 事件因子，作为一路因子喂给 ML
模型，**不直接出预测**。

为什么能平滑接入（不动 Stage 1 已建好的东西）：
- LLMNewsSentiment 继承 BaseFactor，和量价因子是同一个父类、同一个接口。
- 它的 compute() 产出同样的 pd.Series（MultiIndex=(date, ticker)）—— 由 registry 拼到
  FactorFrame 的列；对下游 labels / models / backtest / signals 完全透明。
- 只需在 factors/registry.py 注册一行（默认不启用，因为要 LLM API key + 烧 token）。

实现链路：
  1. 从 panel 推 (ticker, date) 范围 → 经 DataSource.get_news 拿 NewsItem list
  2. 对每条新闻调 LLM 抽 sentiment 分数（缓存命中跳过，烧钱大头）
  3. 聚合到 (ticker, date)：当天新闻情绪用 confidence 加权 mean
  4. reindex 到 panel.index，返回 pd.Series（缺值 NaN）

缓存策略（避免反复调 LLM）：
- 落盘：`data_cache/llm_factor/{ticker}-{date}.json`
- 缓存键：新闻 id（url hash 或 title hash，详见 `_news_id`）
- 缓存内容只存最终情绪分（sentiment / confidence / reason），不存原始 prompt
  + 不存 API key、不存 raw response。
- 同一 news_id 永远不重复调 LLM。

启用方式：
  export ENABLE_LLM_FACTOR=1
  export ANTHROPIC_API_KEY=<your-key>
  # 可选：export LLM_PROVIDER=anthropic  LLM_MODEL=claude-haiku-4-5

默认 ENABLE_LLM_FACTOR=0 —— registry 跳过本因子，pipeline 不会因缺 key 报错。
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

from astock_quant.contracts import NewsItem
from astock_quant.factors.base import BaseFactor
from astock_quant.llm import LLMClient, LLMClientError, NewsSentimentOutput, make_llm_client
from astock_quant.llm.prompts import NEWS_SENTIMENT_SYSTEM, build_news_sentiment_user_prompt

logger = logging.getLogger(__name__)


# ===========================================================================
# 路径 / 缓存常量
# ===========================================================================

# 项目根 = `量化/`（包目录的上一层）
_PKG_ROOT = Path(__file__).resolve().parents[2]
LLM_CACHE_DIR = _PKG_ROOT / "data_cache" / "llm_factor"


# ===========================================================================
# 主因子类
# ===========================================================================

class LLMNewsSentiment(BaseFactor):
    """LLM 新闻情绪因子.

    产出：per-(date, ticker) 的情绪分数（float in [-1, 1]，无新闻 / 调用失败 → NaN）。

    参数：
        lookback:    每个 (ticker, T) 聚合 [T - lookback + 1, T] 这 lookback 日的新闻。
                     默认 1 = 仅当日；3-7 是常见取值（短期记忆 vs 信号稀疏的折中）。
        client:      LLMClient 实现（可注入 mock）。None 时 make_llm_client() 走默认。
        cache_dir:   缓存目录；None 时用包内默认（`data_cache/llm_factor/`）。
        news_fetcher: callable(ticker, start, end) -> list[NewsItem]，用于注入 mock
                     测试或自定义文本源。None 时从 kwargs["news"] / kwargs["news_fetcher"]
                     取（compute 时的运行期注入）。
        max_news_per_day: 单 (ticker, date) 最多调几次 LLM —— 防一天爆发新闻把账户烧光。

    设计取舍：
    - 因子层不强求初始化时就拿到 client —— make_llm_client() 是 lazy 的，
      只在真正要打分（cache miss）时才需要。这样测试场景能注入 mock client。
    - 聚合方式选「confidence 加权 mean」：模型自报的信心越高、对最终分贡献越大；
      没有新闻或全失败 → NaN（不强行填 0）。
    """

    def __init__(
        self,
        *,
        lookback: int = 1,
        client: LLMClient | None = None,
        cache_dir: Path | str | None = None,
        news_fetcher=None,
        max_news_per_day: int = 20,
    ) -> None:
        if lookback < 1:
            raise ValueError(f"lookback 必须 >= 1，得到 {lookback}")
        self.lookback = lookback
        self._client = client
        self._news_fetcher = news_fetcher
        self.cache_dir = Path(cache_dir) if cache_dir else LLM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_news_per_day = max_news_per_day
        # P8 H2：cache 命中统计，每次 compute() 收尾打 debug 日志，方便排查"为什么烧钱多"
        # （nid 用 url 时 fallback=False；缺 url 走 title+content[:500] hash 时 fallback=True）
        self._cache_stats = {"hit": 0, "miss": 0, "fallback_id": 0}

    @property
    def name(self) -> str:
        if self.lookback == 1:
            return "news_sentiment"
        return f"news_sentiment_{self.lookback}d"

    # ----------------------------------------------------------------------
    # compute —— 主入口
    # ----------------------------------------------------------------------

    def compute(self, panel: pd.DataFrame, **kwargs) -> pd.Series:
        """对 panel 计算 LLM 新闻情绪因子.

        kwargs：
            news:         dict[ticker, list[NewsItem]] —— 预拉好的新闻（**测试 / 离线模式优先**，
                          有这个就不调 news_fetcher）。registry 调本因子不会传，但 mock
                          测试和 smoke script 可以直接传。
            news_fetcher: callable(ticker, start, end) -> list[NewsItem] —— 覆盖
                          self._news_fetcher。
        其它 registry 透传的 kwargs（moneyflow / financials）忽略 —— 本因子不用。

        没有 news 数据源 & 没有 fetcher → 全 NaN 返回（不抛异常，符合 BaseFactor 约定）。
        LLM 客户端构造失败（缺 key）→ 全 NaN 返回 + 一次 warning。
        """
        if panel is None or panel.empty:
            return pd.Series(dtype=float, index=panel.index if panel is not None else None,
                             name=self.name)

        # ① 准备新闻源 ----------------------------------------------------
        news_pre: dict[str, list[NewsItem]] = kwargs.get("news") or {}
        news_fetcher = kwargs.get("news_fetcher") or self._news_fetcher

        # ② 准备 LLM client（lazy）---------------------------------------
        client = self._client
        if client is None:
            try:
                client = make_llm_client()
            except LLMClientError as e:
                logger.warning(
                    "LLMNewsSentiment: 构造 LLM client 失败，返回全 NaN: %s", e
                )
                return pd.Series(np.nan, index=panel.index, name=self.name, dtype=float)

        # ③ 对每个 ticker 单独处理 ---------------------------------------
        result_pieces: list[pd.Series] = []
        tickers = panel.index.get_level_values("ticker").unique()
        for raw_ticker in tickers:
            ticker = _normalize_ticker(raw_ticker)
            sub_dates = panel.xs(raw_ticker, level="ticker").index
            if len(sub_dates) == 0:
                continue

            # 决定本 ticker 的新闻清单
            if ticker in news_pre or raw_ticker in news_pre:
                items = news_pre.get(ticker, news_pre.get(raw_ticker, []))
            elif news_fetcher is not None:
                start_d = pd.Timestamp(sub_dates.min()).date().isoformat()
                end_d = pd.Timestamp(sub_dates.max()).date().isoformat()
                try:
                    items = news_fetcher(ticker, start_d, end_d)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "news_fetcher 失败 %s: %s", ticker, e
                    )
                    items = []
            else:
                items = []

            # M1：第三道 look-ahead 防线 —— 丢弃日期晚于 panel 当前 ticker 最大交易日的新闻
            panel_max_date = _to_date(sub_dates.max())
            future_items = [it for it in items if _to_date(it.date) > panel_max_date]
            if future_items:
                logger.warning(
                    "LLMNewsSentiment: 丢弃 %d 条未来新闻 ticker=%s (panel_max=%s, "
                    "future_dates=%s)",
                    len(future_items),
                    ticker,
                    panel_max_date.isoformat(),
                    sorted({_to_date(it.date).isoformat() for it in future_items}),
                )
                items = [it for it in items if _to_date(it.date) <= panel_max_date]

            # ④ 对每个 (ticker, date) 聚合 -------------------------------
            sentiment_by_date = self._aggregate_to_dates(
                ticker=ticker,
                dates=sub_dates,
                news=items,
                client=client,
            )

            # 构造 MultiIndex=(date, raw_ticker) 的 Series 片段
            mi = pd.MultiIndex.from_product(
                [sub_dates, [raw_ticker]],
                names=["date", "ticker"],
            )
            s = pd.Series(
                [sentiment_by_date.get(_to_date(d), np.nan) for d in sub_dates],
                index=mi,
                name=self.name,
                dtype=float,
            )
            result_pieces.append(s)

        if not result_pieces:
            return pd.Series(np.nan, index=panel.index, name=self.name, dtype=float)

        full = pd.concat(result_pieces)
        # reindex 回 panel.index，对齐 registry 的要求
        full = full.reindex(panel.index)
        full.name = self.name
        # P8 H2：本次 compute 的 cache 行为（debug 用，不影响主路径）
        # hit + miss 应近似 = 收到的 news 总条数；fallback_id 比例越高、撞 ID 风险越大
        stats = self._cache_stats
        if stats["hit"] or stats["miss"]:
            logger.debug(
                "LLMNewsSentiment cache: hit=%d miss=%d fallback_id=%d",
                stats["hit"], stats["miss"], stats["fallback_id"],
            )
        return full

    # ----------------------------------------------------------------------
    # 聚合：把单条新闻 LLM 分聚合到 (ticker, date)
    # ----------------------------------------------------------------------

    def _aggregate_to_dates(
        self,
        ticker: str,
        dates: pd.Index,
        news: list[NewsItem],
        client: LLMClient,
    ) -> dict[_date, float]:
        """对每个交易日 T，聚合「过去 lookback 个交易日」的新闻情绪 → 单个 float.

        ⚠️ 窗口口径 = **交易日**（不是自然日）。
        例：lookback=3 + T=周一 → 窗口 = {上周三, 上周四, 上周五, 本周一}（不是 {周六, 周日, 周一}）。
        周末 / 节假日发的新闻：先按 item.date **就近映射到下一个交易日**（周末新闻归并到下周一），
        再按交易日窗口聚合。这样周末新闻不被丢，并且窗口语义稳定。

        聚合方式：confidence 加权 mean。
        没有新闻或全失败 → 该日缺位（dict 里没这个 key → 上层填 NaN）。
        """
        if not news:
            return {}

        # 把交易日排序、建 day_to_pos 索引 —— 为按交易日窗口回溯做准备（O(1) 找下标）
        trading_days = sorted({_to_date(d) for d in dates})
        if not trading_days:
            return {}
        day_to_pos = {d: i for i, d in enumerate(trading_days)}

        # 把新闻按「就近的下一个交易日」分桶
        # （周末 / 节假日新闻归并到下周一 / 节后首日；交易日新闻不变）
        by_trading_day: dict[_date, list[tuple[float, float]]] = {}
        for item in news:
            scored = self._score_news_item(ticker, item, client)
            if scored is None:
                continue
            sent, conf = scored
            news_d = _to_date(item.date)
            bucket = _map_to_next_trading_day(news_d, trading_days)
            if bucket is None:
                # 新闻日期晚于所有 panel 交易日 —— 直接丢弃（防 look-ahead 不会发生，
                # 但稳妥起见这里不归到最后一个 T，避免把未来新闻拽进过去 T 的窗口）
                continue
            by_trading_day.setdefault(bucket, []).append((sent, conf))

        if not by_trading_day:
            return {}

        # 对每个交易日 T，聚合「panel 上 T 往前 lookback 个交易日」窗口内的新闻
        out: dict[_date, float] = {}
        for raw_d in dates:
            t = _to_date(raw_d)
            pos = day_to_pos.get(t)
            if pos is None:
                continue
            # 交易日窗口 = trading_days[pos - lookback + 1 : pos + 1]（左闭右闭，含 T）
            window = trading_days[max(0, pos - self.lookback + 1) : pos + 1]
            samples: list[tuple[float, float]] = []
            for d in window:
                samples.extend(by_trading_day.get(d, []))
            if not samples:
                continue
            # confidence 加权 mean
            weights = np.array([max(c, 1e-3) for _, c in samples])  # 防全 0 confidence
            values = np.array([s for s, _ in samples])
            out[t] = float(np.average(values, weights=weights))
        return out

    # ----------------------------------------------------------------------
    # 单条新闻打分（含缓存）
    # ----------------------------------------------------------------------

    def _score_news_item(
        self,
        ticker: str,
        item: NewsItem,
        client: LLMClient,
    ) -> tuple[float, float] | None:
        """对单条新闻调 LLM 打分（先查缓存，miss 才调 LLM）.

        返回 (sentiment, confidence) 或 None（调用失败 / 信息不足）。
        """
        nid = _news_id(ticker, item)
        # 记 fallback_id：无 url → title+content hash（容易因标题漂移撞 ID 分裂）
        if not item.url:
            self._cache_stats["fallback_id"] += 1
        cached = self._load_cache(ticker, item.date, nid)
        if cached is not None:
            self._cache_stats["hit"] += 1
            return cached["sentiment"], cached["confidence"]
        self._cache_stats["miss"] += 1

        # cache miss → 调 LLM
        user = build_news_sentiment_user_prompt(
            ticker=ticker,
            title=item.title,
            content=item.content,
            source=item.source,
            date=item.date.isoformat(),
        )
        try:
            out: NewsSentimentOutput = client.chat_json(
                messages=[{"role": "user", "content": user}],
                schema=NewsSentimentOutput,
                system=NEWS_SENTIMENT_SYSTEM,
                temperature=0.0,
                # 768 token：防中文 reason 字段被截断导致 JSON 解析失败。
                # 中文约 1.5 char/tok，768 给 reason 留 ~500 字符，复杂理由也够用。
                # 历史：256 时 35% 被截断（P7 Step 1）→ 升 512 后剩 5-10%（P7 v2 实测）→
                # 升 768 收紧最后这部分（P7-next-3）。不拉到 1024：边际收益递减，
                # 多余 padding 按 token 计费浪费。
                max_tokens=768,
            )
        except LLMClientError as e:
            logger.warning("LLM 打分失败（跳过）ticker=%s title=%r: %s",
                           ticker, item.title[:30], e)
            return None

        # 截断到 [-1, 1]，防 LLM 越界
        sent = float(np.clip(out.sentiment, -1.0, 1.0))
        conf = float(np.clip(out.confidence, 0.0, 1.0))

        # 写缓存（不存 API key / 原始 prompt / raw response）
        self._save_cache(ticker, item.date, nid, {
            "sentiment": sent,
            "confidence": conf,
            "reason": out.reason[:200] if out.reason else "",
            "title": item.title[:120],
            "source": item.source,
        })
        return sent, conf

    # ----------------------------------------------------------------------
    # 缓存读写
    # ----------------------------------------------------------------------

    def _cache_path(self, ticker: str, news_date: _date) -> Path:
        d = _to_date(news_date)
        return self.cache_dir / f"{ticker}-{d.isoformat()}.json"

    def _load_cache(
        self, ticker: str, news_date: _date, news_id: str,
    ) -> dict | None:
        path = self._cache_path(ticker, news_date)
        if not path.exists():
            return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("缓存读取失败 %s: %s", path, e)
            return None
        entry = blob.get(news_id)
        if not entry:
            return None
        # 简单 schema 校验
        if not all(k in entry for k in ("sentiment", "confidence")):
            return None
        return entry

    def _save_cache(
        self, ticker: str, news_date: _date, news_id: str, entry: dict,
    ) -> None:
        path = self._cache_path(ticker, news_date)
        try:
            blob = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists() else {}
            )
        except (OSError, json.JSONDecodeError):
            blob = {}
        blob[news_id] = entry
        try:
            path.write_text(
                json.dumps(blob, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("缓存写入失败 %s: %s", path, e)


# ===========================================================================
# helpers
# ===========================================================================

def _news_id(ticker: str, item: NewsItem) -> str:
    """新闻唯一 id：优先用 url，否则 hash(ticker + date + title + content[:500]).

    fallback 路径加 content[:500] 是 P8 H2 修法：仅靠 title 容易被两种情况撞 ID 分裂：
    - 同事件不同来源标题略不同（"茅台业绩超预期" vs "贵州茅台一季报亮眼"）→ 视为两条 → 重复打分浪费 token
    - 标题里带时间戳 / 排名（"06:30 早盘速递 …"）→ 同条新闻不同时刻拉取 ID 漂移 → 缓存失效

    把 content 前 500 字符纳入 hash：长度限上界防 hash 抖动，区分度对短摘要 / 详细全文都够。
    """
    if item.url:
        # 用 url 的稳定 hash（防奇怪字符 / 太长）—— 优先路径，content 不参与
        return "u:" + hashlib.sha1(item.url.encode("utf-8")).hexdigest()[:16]
    content_head = (item.content or "")[:500]
    raw = f"{ticker}|{item.date.isoformat()}|{item.title}|{content_head}"
    return "t:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _map_to_next_trading_day(news_d: _date, trading_days: list[_date]) -> _date | None:
    """把新闻自然日 → 它「下一个」交易日（>= news_d 的最小 trading_day）.

    用 bisect_left 在 sorted trading_days 里找位置：
    - news_d 本身是交易日 → 返回 news_d
    - news_d 是周末 / 节假日 → 返回紧随其后的交易日（周末新闻归并到下周一 / 节后首日）
    - news_d 晚于所有 trading_days（未来新闻）→ 返回 None，调用方丢弃

    复杂度 O(log n)，对 1000+ 交易日 panel 仍 sub-µs 级。
    """
    if not trading_days:
        return None
    idx = bisect.bisect_left(trading_days, news_d)
    if idx >= len(trading_days):
        return None
    return trading_days[idx]


def _normalize_ticker(t) -> str:
    """panel 的 ticker 可能是 int（CSV roundtrip 丢前导 0）或 str；统一成 6 位字符串.

    例：858 → "000858"；"600519" → "600519"。
    """
    s = str(t).strip()
    # 数字纯整型 → 左 pad
    if s.isdigit():
        return s.zfill(6)
    return s


def _to_date(d) -> _date:
    """把各种 date-like（datetime / pd.Timestamp / str / date）统一成 date。"""
    if isinstance(d, _date) and not isinstance(d, pd.Timestamp):
        # date 但不是 Timestamp 的子类
        return d
    return pd.Timestamp(d).date()


# ===========================================================================
# 对外 export
# ===========================================================================

__all__ = ["LLMNewsSentiment"]
