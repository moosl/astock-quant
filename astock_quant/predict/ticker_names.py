"""ticker → 中文名映射。

- STAGE1_NAMES：30 只蓝筹硬编码，覆盖 Stage 1 universe
- STAGE1_SHORT_NAMES：大白话段用的简称
- get_ticker_name(code)：先查硬编码，再查缓存 JSON，最后兜底调 akshare
- get_ticker_short_name(code)：同款，返回简称
"""

from __future__ import annotations

import json
from pathlib import Path

STAGE1_NAMES: dict[str, str] = {
    # 食品饮料 3
    "600519": "贵州茅台",
    "000858": "五粮液",
    "600887": "伊利股份",
    # 银行 3
    "601398": "工商银行",
    "600036": "招商银行",
    "000001": "平安银行",
    # 非银金融 3
    "601318": "中国平安",
    "600030": "中信证券",
    "300059": "东方财富",
    # 新能源 3
    "300750": "宁德时代",
    "002594": "比亚迪",
    "601012": "隆基绿能",
    # 医药生物 3
    "600276": "恒瑞医药",
    "300760": "迈瑞医疗",
    "603259": "药明康德",
    # 家电 3
    "000333": "美的集团",
    "000651": "格力电器",
    "600690": "海尔智家",
    # 科技/电子 4
    "002415": "海康威视",
    "002475": "立讯精密",
    "000725": "京东方A",
    "600703": "三安光电",
    # 资源/周期 3
    "601899": "紫金矿业",
    "600028": "中国石化",
    "601088": "中国神华",
    # 基建/地产/交运 2
    "601668": "中国建筑",
    "600009": "上海机场",
    # 汽车/机械 2
    "601633": "长城汽车",
    "600031": "三一重工",
    # 消费/零售 1
    "603288": "海天味业",
}

STAGE1_SHORT_NAMES: dict[str, str] = {
    "600519": "茅台",
    "000858": "五粮液",
    "600887": "伊利",
    "601398": "工行",
    "600036": "招行",
    "000001": "平安",
    "601318": "中国平安",
    "600030": "中信证",
    "300059": "东财",
    "300750": "宁德",
    "002594": "比亚迪",
    "601012": "隆基",
    "600276": "恒瑞",
    "300760": "迈瑞",
    "603259": "药明",
    "000333": "美的",
    "000651": "格力",
    "600690": "海尔",
    "002415": "海康",
    "002475": "立讯",
    "000725": "京东方",
    "600703": "三安",
    "601899": "紫金",
    "600028": "中石化",
    "601088": "神华",
    "601668": "中建",
    "600009": "上海机场",
    "601633": "长城",
    "600031": "三一",
    "603288": "海天",
}

_CACHE_PATH: Path | None = None


def _cache_path() -> Path:
    global _CACHE_PATH
    if _CACHE_PATH is None:
        try:
            from astock_quant.config.settings import SETTINGS
            _CACHE_PATH = Path(SETTINGS.data_cache_dir) / "ticker_names_cache.json"
        except Exception:
            _CACHE_PATH = Path("data_cache") / "ticker_names_cache.json"
    return _CACHE_PATH


def _load_cache() -> dict[str, str]:
    p = _cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _fetch_from_akshare() -> dict[str, str]:
    import akshare as ak
    df = ak.stock_info_a_code_name()
    return dict(zip(df["code"].astype(str).str.zfill(6), df["name"]))


def get_ticker_name(code: str) -> str:
    """ticker → 中文全名，3 道 fallback：硬编码 → cache → akshare → code 本身."""
    code = str(code).strip()
    if code in STAGE1_NAMES:
        return STAGE1_NAMES[code]

    cached = _load_cache()
    if code in cached:
        return cached[code]

    try:
        name_map = _fetch_from_akshare()
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(name_map, ensure_ascii=False), encoding="utf-8")
        return name_map.get(code, code)
    except Exception:
        return code


def get_ticker_short_name(code: str) -> str:
    """ticker → 大白话简称，fallback 到全名再到 code 本身."""
    code = str(code).strip()
    if code in STAGE1_SHORT_NAMES:
        return STAGE1_SHORT_NAMES[code]
    full = get_ticker_name(code)
    return full if full != code else code
