"""LLM 个股价值分析层 —— 调用 Codex CLI 写质性解读 + 市场综述.

设计思路:
- 选股仍由 value_score.py 数学公式(诚信红线), 这里只写「人话解读」.
- SKILL adapter (astock_data_skill) 拉 6-8 个端点, 数据 → prompt → Codex CLI → markdown.
- 失败兜底: 端点失败返回空, LLM 挂返回 fallback dict (llm_rationale=None).
- 所有输出含「AI 生成 / 不构成投资建议」disclaimer.

公开 API:
    analyze_stock(ticker, *, perspective, depth, factor_context, pre_fetched) -> dict
    market_overview(picks_summary, *, perspective) -> dict
"""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ===========================================================================
# Disclaimer (诚信红线) —— 每段 LLM 输出末尾追加
# ===========================================================================

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **本分析为 AI 生成 (Codex CLI), 基于公开数据自动整理, 不构成投资建议。** "
    "选股仍由数学公式打分, AI 仅写解读。回测跑赢不等于实盘赚钱。"
)


# ===========================================================================
# 价值投资视角的 system prompt
# ===========================================================================

VALUE_SYSTEM_PROMPT = """你是一位长期价值投资分析师, 风格类似巴菲特/芒格/段永平,
看公司不看短期股价。任务: 根据提供的真实数据 (估值/财务/资金流/新闻/分红),
对个股做一段价值视角的质性诊断。

【分析框架】(按顺序写)
1. 估值水平 —— PE/PB 是高是低? 跟历史比、跟同行比?
2. 财务质量 —— ROE、毛利、净利、负债、自由现金流, 有没有真本事赚钱?
3. 长期盈利能力 —— 近 5 年净利润 CAGR、分红连续性、ROE 稳定性
4. 安全边际 —— 当前价格离合理估值多远? 万一行业变差能扛吗?
5. 风险点 —— 行业周期、估值陷阱、单一客户依赖、政策风险、解禁压力
6. 综合诊断 —— 一句话: 这是不是个能让人睡得着觉的标的?

【硬要求】
- **大白话**, 避免行话 (PEG/EV/EBITDA 等术语必须用中文+举例解释)
- 看到数据矛盾要点出来 (e.g. ROE 高但负债也高 → 加了杠杆)
- 警惕「价值陷阱」: 便宜但行业在死, 增长停滞, 治理差
- 警惕「靠单年」: 只看一年好是噪音, 要看 3-5 年趋势
- 警惕「行业过度集中」: 银行/地产 ROE 高但同质化严重
- 数据缺失/异常的字段, 显式说"数据缺失"或"该项异常", 不要编
- **不出现「买入/卖出/建议」字眼**, 改用「这家公司的特点是...」「需要关注的是...」

【输出格式】
- 纯 markdown, 无 H1
- summary 模式约 150-250 字
- full 模式约 500-700 字, 分小节
- 末尾自然结尾, 不写 "## 总结" 之类的标题
"""

GENERAL_SYSTEM_PROMPT = """你是一位 A 股研究分析师。根据提供的真实数据 (估值/财务/资金流/新闻),
对个股做一段中性、客观的概况分析。大白话, 不堆术语, 不给买卖建议, 末尾自然结尾。
约 200-300 字 markdown。
"""

MULTI_SYSTEM_PROMPT = """你是一位 A 股投资研究员。基于真实数据从「价值」「成长」「技术面/资金面」
三个视角分别写一小段 (各 100-150 字), 然后用一句话给出综合特征 (不是买卖建议)。
大白话, markdown, 各视角用 ### 小标题。
"""

MARKET_SYSTEM_PROMPT = """你是一位 A 股市场策略师。根据提供的当日数据 (强势股、北向资金、龙虎榜、
价值选股名单),写一段 150-250 字的市场速览。

【写作要求】
- 大白话, 不堆术语
- 指出今天市场关心什么主题 (从强势股 reason tags 看)
- 资金动向: 北向是流入还是流出, 龙虎榜活跃度
- 价值选股名单的行业分布特点 (e.g. 偏银行 / 偏消费 / 偏科技)
- 末尾用一句话点出今天最值得记住的一件事
- 不出现买卖建议
"""


# ===========================================================================
# Codex CLI client (延迟实例化, 测试可注入 mock)
# ===========================================================================


def _make_client():
    """创建 Codex CLI 客户端. 单独函数方便测试 monkeypatch."""
    from astock_quant.llm.client import make_llm_client
    return make_llm_client("codex")


# ===========================================================================
# 端点并行拉取
# ===========================================================================


def _fetch_endpoints_for_value(ticker: str) -> dict[str, Any]:
    """价值视角拉数据: 行情/财务/分红/股东户数/资金流/新闻.

    用 ThreadPoolExecutor 并行 6 个端点, 每个 timeout 自带 (SKILL adapter 内部),
    顶层加一个 30s 总超时兜底. 任何端点失败返回该 key=空容器, 不抛崩.
    """
    from astock_data_skill import (
        dividend_history,
        eastmoney_stock_info,
        eastmoney_stock_news,
        holder_num_change,
        stock_fund_flow_120d,
        tencent_quote,
    )

    tasks: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            "quote":      ex.submit(tencent_quote, [ticker]),
            "info":       ex.submit(eastmoney_stock_info, ticker),
            "dividends":  ex.submit(dividend_history, ticker, 10),
            "holders":    ex.submit(holder_num_change, ticker, 6),
            "fund_flow":  ex.submit(stock_fund_flow_120d, ticker),
            "news":       ex.submit(eastmoney_stock_news, ticker, 10),
        }
        for name, fut in futs.items():
            try:
                tasks[name] = fut.result(timeout=20)
            except Exception as e:  # noqa: BLE001
                logger.warning("fetch %s for %s failed: %s", name, ticker, e)
                tasks[name] = {} if name in ("quote", "info") else []
    return tasks


def _fetch_endpoints_for_market() -> dict[str, Any]:
    """市场速览拉数据: 强势股 + 北向 + 全市场龙虎榜."""
    from astock_data_skill import (
        daily_dragon_tiger,
        hsgt_realtime,
        ths_hot_reason,
    )

    tasks: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {
            "hot":     ex.submit(ths_hot_reason),
            "north":   ex.submit(hsgt_realtime),
            "dragon":  ex.submit(daily_dragon_tiger),
        }
        for name, fut in futs.items():
            try:
                tasks[name] = fut.result(timeout=15)
            except Exception as e:  # noqa: BLE001
                logger.warning("fetch market %s failed: %s", name, e)
                tasks[name] = None
    return tasks


# ===========================================================================
# 数据 → prompt 输入文本
# ===========================================================================


def _build_value_context(
    ticker: str,
    factor_context: dict | None,
    endpoints: dict[str, Any],
) -> str:
    """把抓到的端点数据 + factor_context (value_score 因子)拼成 prompt 输入.

    输出是一段中文 markdown 文本喂给 LLM. 缺数据的字段写「-」不编造.
    """
    lines: list[str] = []
    lines.append(f"## 个股数据 - {ticker}")

    # 因子上下文 (来自 value_score)
    if factor_context:
        lines.append("\n### 综合打分 (来自项目 value_score 数学公式)")
        comp = factor_context.get("composite_score")
        v = factor_context.get("value_score")
        q = factor_context.get("quality_score")
        g = factor_context.get("growth_score")
        if comp is not None:
            lines.append(f"- 综合分: {comp:.3f}")
        if v is not None:
            lines.append(f"- 估值分位 (越高越便宜): {v:.3f}")
        if q is not None:
            lines.append(f"- 质量分位 (越高越赚钱): {q:.3f}")
        if g is not None:
            lines.append(f"- 成长分位: {g:.3f}")

    # 行情 + 基本面
    q = (endpoints.get("quote") or {}).get(ticker) or {}
    info = endpoints.get("info") or {}
    lines.append("\n### 估值 + 基本面")
    lines.append(f"- 名称: {q.get('name') or info.get('name') or '-'}")
    lines.append(f"- 行业: {info.get('industry') or '-'}")
    lines.append(f"- 当前价: {q.get('price', '-')} 元")
    lines.append(f"- PE(TTM): {q.get('pe_ttm', '-')}")
    lines.append(f"- PB: {q.get('pb', '-')}")
    lines.append(f"- 总市值: {q.get('mcap_yi', '-')} 亿")
    lines.append(f"- 换手率: {q.get('turnover_pct', '-')}%")
    list_date = info.get("list_date") or "-"
    if list_date and len(list_date) == 8:
        list_date = f"{list_date[:4]}-{list_date[4:6]}-{list_date[6:]}"
    lines.append(f"- 上市日期: {list_date}")

    # ROE (从 factor_context 拿, TTM 全年口径)
    roe = factor_context.get("roe") if factor_context else None
    if roe is not None:
        lines.append(f"- ROE (TTM 全年口径): {roe:.2f}%")

    # 分红
    divs = endpoints.get("dividends") or []
    if divs:
        recent = divs[:5]
        lines.append("\n### 近 5 期分红")
        for d in recent:
            lines.append(
                f"- {d.get('date', '?')}: 每股派息 {d.get('bonus_rmb', 0)} 元"
                f", 送股 {d.get('bonus_ratio', 0)}, 转增 {d.get('transfer_ratio', 0)}"
            )
    else:
        lines.append("\n### 近期分红: 暂无记录")

    # 股东户数变化
    holders = endpoints.get("holders") or []
    if holders:
        lines.append("\n### 股东户数变化 (近 4 期)")
        for h in holders[:4]:
            lines.append(
                f"- {h.get('date', '?')}: 股东数 {h.get('holder_num', 0)},"
                f" 环比变化 {h.get('change_ratio', 0)}%"
            )

    # 资金流近 20 日
    ff = endpoints.get("fund_flow") or []
    if ff:
        recent = ff[-20:]
        total_main = sum(d.get("main_net", 0) for d in recent)
        lines.append("\n### 近 20 日资金流向")
        lines.append(f"- 主力累计净流入: {total_main / 1e8:.2f} 亿")

    # 新闻 (取标题, 截前 5 条)
    news = endpoints.get("news") or []
    if news:
        lines.append("\n### 近期新闻 (前 5 条)")
        for n in news[:5]:
            t = (n.get("time", "") or "")[:10]
            title = n.get("title", "")
            lines.append(f"- {t}: {title}")

    return "\n".join(lines)


def _build_market_context(endpoints: dict[str, Any], picks_summary: list[dict]) -> str:
    """市场速览 prompt 输入: 强势股 / 北向 / 龙虎榜 + Top 5 价值选股."""
    lines: list[str] = ["## 今日市场数据"]

    # 强势股 reason 词频
    hot = endpoints.get("hot")
    if hot is not None and hasattr(hot, "empty") and not hot.empty:
        from collections import Counter
        col = "题材归因" if "题材归因" in hot.columns else "reason"
        if col in hot.columns:
            all_tags: list[str] = []
            for r in hot[col].dropna().head(50):
                tags = [t.strip() for t in str(r).split("+") if t.strip()]
                all_tags.extend(tags)
            cnt = Counter(all_tags).most_common(8)
            if cnt:
                lines.append("\n### 当日 TOP 题材热度")
                for tag, n in cnt:
                    lines.append(f"- {tag}: {n} 只")

    # 北向资金
    north = endpoints.get("north")
    if north is not None and hasattr(north, "empty") and not north.empty:
        try:
            valid = north.dropna()
            if not valid.empty:
                last = valid.iloc[-1]
                lines.append("\n### 北向资金 (累计净买入)")
                lines.append(f"- 沪股通: {last.get('hgt_yi', '-')} 亿")
                lines.append(f"- 深股通: {last.get('sgt_yi', '-')} 亿")
        except Exception:  # noqa: BLE001
            pass

    # 龙虎榜
    dragon = endpoints.get("dragon")
    if dragon and dragon.get("total_records"):
        lines.append(f"\n### 全市场龙虎榜: 共 {dragon['total_records']} 只上榜")
        for s in (dragon.get("stocks") or [])[:5]:
            lines.append(
                f"- {s.get('code', '')} {s.get('name', '')}: "
                f"净买 {s.get('net_buy_wan', 0)} 万, 涨跌 {s.get('change_pct', 0)}%"
            )

    # Top 5 价值选股
    if picks_summary:
        lines.append("\n### 本期价值选股 Top 5 (来自项目 value_score 数学公式)")
        for i, p in enumerate(picks_summary[:5], 1):
            name = p.get("name") or p.get("ticker", "?")
            score = p.get("composite_score", 0)
            try:
                s_str = f"{float(score):.3f}"
            except (TypeError, ValueError):
                s_str = str(score)
            lines.append(f"- #{i} {p.get('ticker', '?')} {name}: 综合分 {s_str}")

    return "\n".join(lines)


# ===========================================================================
# 公开 API: analyze_stock
# ===========================================================================


def _pick_system(perspective: str) -> str:
    if perspective == "value":
        return VALUE_SYSTEM_PROMPT
    if perspective == "multi":
        return MULTI_SYSTEM_PROMPT
    return GENERAL_SYSTEM_PROMPT


def _max_tokens_for(depth: str) -> int:
    return 1500 if depth == "full" else 700


def analyze_stock(
    ticker: str,
    *,
    perspective: str = "value",
    depth: str = "summary",
    factor_context: dict | None = None,
    pre_fetched: dict | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """对单只票生成 LLM 质性分析.

    Args:
        ticker:         股票代码 (会被 normalize)
        perspective:    "value" / "general" / "multi"
        depth:          "summary" (~200字) / "full" (~600字)
        factor_context: value_score 因子 dict (composite_score / value_score /
                        quality_score / growth_score / pe / pb / roe)
        pre_fetched:    预拉的 endpoints dict, 避免重复请求 (daily 批量场景用)
        client:         可注入 mock LLM client (测试用); None 时走 make_llm_client.

    Returns:
        {ticker, name, markdown, fetched_endpoints, tokens_used, generated_at}.
        失败兜底: markdown=None, 其它字段尽量填.
    """
    from astock_data_skill import normalize_ticker

    code = normalize_ticker(ticker)
    generated_at = _dt.datetime.now().isoformat(timespec="seconds")

    # 拿数据
    endpoints = pre_fetched if pre_fetched is not None else _fetch_endpoints_for_value(code)
    quote = (endpoints.get("quote") or {}).get(code) or {}
    info = endpoints.get("info") or {}
    name = quote.get("name") or info.get("name") or ""

    # 拼 prompt 输入
    context = _build_value_context(code, factor_context, endpoints)
    system = _pick_system(perspective)
    user = (
        f"以下是 {name or code} ({code}) 的真实数据, 请按系统消息写"
        f"{'详细' if depth == 'full' else '简洁'}的{('价值投资' if perspective == 'value' else '')}分析:\n\n"
        f"{context}"
    )

    # 调 LLM
    if client is None:
        try:
            client = _make_client()
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM client init failed for %s: %s", code, e)
            return _fallback_result(code, name, endpoints, generated_at,
                                    err=f"LLM 客户端初始化失败: {e}")

    try:
        resp = client.chat(
            [{"role": "user", "content": user}],
            system=system,
            temperature=0.3,
            max_tokens=_max_tokens_for(depth),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM call failed for %s: %s", code, e)
        return _fallback_result(code, name, endpoints, generated_at,
                                err=f"LLM 调用失败: {e}")

    markdown = (resp.content or "").strip()
    if not markdown:
        return _fallback_result(code, name, endpoints, generated_at,
                                err="LLM 返回空内容")
    markdown += DISCLAIMER

    tokens_used = 0
    if resp.usage:
        tokens_used = int(resp.usage.get("input_tokens", 0)) + int(
            resp.usage.get("output_tokens", 0)
        )

    return {
        "ticker": code,
        "name": name,
        "markdown": markdown,
        "fetched_endpoints": list(endpoints.keys()),
        "tokens_used": tokens_used,
        "generated_at": generated_at,
        "perspective": perspective,
        "depth": depth,
    }


def market_overview(
    picks_summary: list[dict] | None = None,
    *,
    perspective: str = "value",
    client: Any = None,
) -> dict[str, Any]:
    """市场速览 LLM: 综合强势股 + 北向 + 龙虎榜 + Top 5 picks 写 200 字综述.

    Returns:
        {markdown, fetched_endpoints, tokens_used, generated_at}. 失败 markdown=None.
    """
    _ = perspective  # 保留参数, 当前只有市场视角
    generated_at = _dt.datetime.now().isoformat(timespec="seconds")
    endpoints = _fetch_endpoints_for_market()
    context = _build_market_context(endpoints, picks_summary or [])

    if client is None:
        try:
            client = _make_client()
        except Exception as e:  # noqa: BLE001
            logger.warning("market LLM init failed: %s", e)
            return {
                "markdown": None,
                "fetched_endpoints": list(endpoints.keys()),
                "tokens_used": 0,
                "generated_at": generated_at,
                "error": str(e),
            }

    try:
        resp = client.chat(
            [{"role": "user", "content": context}],
            system=MARKET_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=600,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("market LLM call failed: %s", e)
        return {
            "markdown": None,
            "fetched_endpoints": list(endpoints.keys()),
            "tokens_used": 0,
            "generated_at": generated_at,
            "error": str(e),
        }

    md = (resp.content or "").strip()
    if not md:
        return {
            "markdown": None,
            "fetched_endpoints": list(endpoints.keys()),
            "tokens_used": 0,
            "generated_at": generated_at,
            "error": "LLM 返回空内容",
        }
    md += DISCLAIMER

    tokens_used = 0
    if resp.usage:
        tokens_used = int(resp.usage.get("input_tokens", 0)) + int(
            resp.usage.get("output_tokens", 0)
        )

    return {
        "markdown": md,
        "fetched_endpoints": list(endpoints.keys()),
        "tokens_used": tokens_used,
        "generated_at": generated_at,
    }


# ===========================================================================
# Helpers
# ===========================================================================


def _fallback_result(
    code: str,
    name: str,
    endpoints: dict,
    generated_at: str,
    *,
    err: str,
) -> dict[str, Any]:
    """LLM 调用失败时的降级返回. markdown=None 让上层用旧 reason."""
    return {
        "ticker": code,
        "name": name,
        "markdown": None,
        "fetched_endpoints": list(endpoints.keys()) if endpoints else [],
        "tokens_used": 0,
        "generated_at": generated_at,
        "error": err,
    }


__all__ = [
    "analyze_stock",
    "market_overview",
    "DISCLAIMER",
    "VALUE_SYSTEM_PROMPT",
    "MARKET_SYSTEM_PROMPT",
]
