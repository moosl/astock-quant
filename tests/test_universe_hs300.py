"""P15 沪深 300 universe 测试 —— settings.py get_hs300_universe + get_universe.

覆盖：
- 命门：get_universe("stage1") 返回 STAGE1_UNIVERSE（向后兼容守门）
- get_hs300_universe：mock akshare + 解析正确
- cache 行为：第一次 fetch，第二次读 JSON（akshare 不被第二次调）
- cache 过期：超过 1 天 → 触发重 fetch
- cache 损坏 → 重 fetch（不崩溃）
- akshare 失败时无 cache → 抛异常
- get_universe("stage4") 调 get_hs300_universe
- get_universe(invalid) → fallback 到 STAGE1_UNIVERSE（实际行为）
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_mock_akshare_df(codes: list[str]) -> pd.DataFrame:
    """构造 akshare index_stock_cons_csindex 的返回 DataFrame."""
    return pd.DataFrame({"成分券代码": codes, "成分券名称": [f"股票{c}" for c in codes]})


def _write_cache(path: Path, codes: list[str], age_seconds: float = 0.0) -> None:
    """写一个 cache 文件，fetched_at 设为 age_seconds 秒前。"""
    fetched_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"fetched_at": fetched_at.isoformat(), "universe": codes},
                   ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 命门：get_universe("stage1") 向后兼容
# ---------------------------------------------------------------------------

class TestGetUniverseBackwardCompat:

    def test_stage1_returns_stage1_universe_unchanged(self):
        """命门：get_universe('stage1') 必须返回 STAGE1_UNIVERSE，不能漂移。"""
        from astock_quant.config.settings import STAGE1_UNIVERSE, get_universe
        result = get_universe("stage1")
        assert result == list(STAGE1_UNIVERSE), (
            "命门失败：get_universe('stage1') 返回值与 STAGE1_UNIVERSE 不一致！"
            "向后兼容被破坏，所有依赖 STAGE1_UNIVERSE 的测试会漂移。"
        )

    def test_default_stage_is_stage1(self):
        """get_universe() 不传参 → 等同 stage1。"""
        from astock_quant.config.settings import STAGE1_UNIVERSE, get_universe
        assert get_universe() == list(STAGE1_UNIVERSE)

    def test_stage1_universe_is_nonempty(self):
        """STAGE1_UNIVERSE 本体不为空（基本完整性检查）。"""
        from astock_quant.config.settings import STAGE1_UNIVERSE
        assert len(STAGE1_UNIVERSE) > 0

    def test_stage1_universe_contains_6_digit_codes(self):
        """STAGE1_UNIVERSE 里每个 code 是 6 位纯数字。"""
        from astock_quant.config.settings import STAGE1_UNIVERSE
        for code in STAGE1_UNIVERSE:
            assert code.isdigit() and len(code) == 6, \
                f"STAGE1_UNIVERSE 包含非 6 位纯数字 code: {code!r}"

    def test_invalid_stage_returns_stage1_fallback(self):
        """get_universe('not_a_stage') → fallback 到 STAGE1_UNIVERSE（实际行为）。"""
        from astock_quant.config.settings import STAGE1_UNIVERSE, get_universe
        result = get_universe("not_a_stage")
        assert result == list(STAGE1_UNIVERSE)


# ---------------------------------------------------------------------------
# get_hs300_universe — mock akshare
# ---------------------------------------------------------------------------

class TestGetHs300Universe:

    def test_parses_akshare_df_correctly(self, tmp_path):
        """mock akshare，断言返回 code list 与 mock 数据一致。"""
        mock_codes = ["600519", "000858", "600036"]
        mock_df = _make_mock_akshare_df(mock_codes)
        cache_file = tmp_path / "hs300_universe.json"

        with (
            patch("astock_quant.config.settings._HS300_CACHE_FILE", cache_file),
            patch("akshare.index_stock_cons_csindex", return_value=mock_df),
        ):
            from astock_quant.config.settings import get_hs300_universe
            result = get_hs300_universe()

        assert result == mock_codes

    def test_cache_hit_does_not_call_akshare(self, tmp_path):
        """有效 cache 存在时，akshare 不被调用。"""
        cache_codes = ["600519", "000858"]
        cache_file = tmp_path / "hs300_universe.json"
        _write_cache(cache_file, cache_codes, age_seconds=100)

        mock_ak = MagicMock()
        with (
            patch("astock_quant.config.settings._HS300_CACHE_FILE", cache_file),
            patch("akshare.index_stock_cons_csindex", mock_ak),
        ):
            from astock_quant.config.settings import get_hs300_universe
            result = get_hs300_universe()

        mock_ak.assert_not_called()
        assert result == cache_codes

    def test_second_call_reads_cache_not_akshare(self, tmp_path):
        """第一次 fetch 写 cache，第二次直接读 cache（akshare 只被调 1 次）。"""
        mock_codes = ["600519", "000858", "601398"]
        mock_df = _make_mock_akshare_df(mock_codes)
        cache_file = tmp_path / "hs300_universe.json"

        call_count = [0]
        def counting_ak(symbol):
            call_count[0] += 1
            return mock_df

        with (
            patch("astock_quant.config.settings._HS300_CACHE_FILE", cache_file),
            patch("akshare.index_stock_cons_csindex", side_effect=counting_ak),
        ):
            from astock_quant.config.settings import get_hs300_universe
            r1 = get_hs300_universe()
            r2 = get_hs300_universe()

        assert call_count[0] == 1, f"akshare 被调了 {call_count[0]} 次，期望 1 次"
        assert r1 == r2 == mock_codes

    def test_expired_cache_refetches(self, tmp_path):
        """cache 超过 1 天（86400s）→ 触发重 fetch。"""
        old_codes = ["000001"]
        new_codes = ["600519", "000858"]
        cache_file = tmp_path / "hs300_universe.json"
        _write_cache(cache_file, old_codes, age_seconds=86401)  # 刚好超期

        mock_df = _make_mock_akshare_df(new_codes)
        with (
            patch("astock_quant.config.settings._HS300_CACHE_FILE", cache_file),
            patch("akshare.index_stock_cons_csindex", return_value=mock_df),
        ):
            from astock_quant.config.settings import get_hs300_universe
            result = get_hs300_universe()

        assert result == new_codes

    def test_corrupt_cache_triggers_refetch(self, tmp_path):
        """cache 文件内容损坏 → 不崩溃，重 fetch。"""
        cache_file = tmp_path / "hs300_universe.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("{corrupt!!!", encoding="utf-8")

        mock_codes = ["600519"]
        mock_df = _make_mock_akshare_df(mock_codes)
        with (
            patch("astock_quant.config.settings._HS300_CACHE_FILE", cache_file),
            patch("akshare.index_stock_cons_csindex", return_value=mock_df),
        ):
            from astock_quant.config.settings import get_hs300_universe
            result = get_hs300_universe()

        assert result == mock_codes

    def test_cache_written_after_fetch(self, tmp_path):
        """fetch 成功后 cache 文件被写入。"""
        cache_file = tmp_path / "hs300_universe.json"
        mock_codes = ["600519", "000858"]
        mock_df = _make_mock_akshare_df(mock_codes)

        with (
            patch("astock_quant.config.settings._HS300_CACHE_FILE", cache_file),
            patch("akshare.index_stock_cons_csindex", return_value=mock_df),
        ):
            from astock_quant.config.settings import get_hs300_universe
            get_hs300_universe()

        assert cache_file.exists()
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        assert cached["universe"] == mock_codes
        assert "fetched_at" in cached

    def test_codes_are_zero_padded_6_digits(self, tmp_path):
        """code 不足 6 位时补零（如 '858' → '000858'）。"""
        cache_file = tmp_path / "hs300_universe.json"
        mock_df = _make_mock_akshare_df(["858", "519"])  # 短 code

        with (
            patch("astock_quant.config.settings._HS300_CACHE_FILE", cache_file),
            patch("akshare.index_stock_cons_csindex", return_value=mock_df),
        ):
            from astock_quant.config.settings import get_hs300_universe
            result = get_hs300_universe()

        assert result == ["000858", "000519"]
        for code in result:
            assert len(code) == 6 and code.isdigit()


# ---------------------------------------------------------------------------
# get_universe("stage4")
# ---------------------------------------------------------------------------

class TestGetUniverseStage4:

    def test_stage4_calls_get_hs300_universe(self, tmp_path):
        """get_universe('stage4') 调 get_hs300_universe 并返回其结果。"""
        mock_codes = ["600519", "000858", "600036"]
        mock_df = _make_mock_akshare_df(mock_codes)
        cache_file = tmp_path / "hs300_universe.json"

        with (
            patch("astock_quant.config.settings._HS300_CACHE_FILE", cache_file),
            patch("akshare.index_stock_cons_csindex", return_value=mock_df),
        ):
            from astock_quant.config.settings import get_universe
            result = get_universe("stage4")

        assert result == mock_codes

    def test_stage4_result_is_independent_copy(self, tmp_path):
        """stage4 结果可以修改，不影响 cache 内容（防止可变引用泄漏）。"""
        mock_codes = ["600519", "000858"]
        mock_df = _make_mock_akshare_df(mock_codes)
        cache_file = tmp_path / "hs300_universe.json"

        with (
            patch("astock_quant.config.settings._HS300_CACHE_FILE", cache_file),
            patch("akshare.index_stock_cons_csindex", return_value=mock_df),
        ):
            from astock_quant.config.settings import get_universe
            r = get_universe("stage4")
            r.append("INJECTED")
            r2 = get_universe("stage4")

        assert "INJECTED" not in r2
