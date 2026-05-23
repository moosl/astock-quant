"""astock_data_skill 包的单元 + 集成测试.

Mock HTTP 层 (requests.get/post + urllib) 验证: 端点能 import / 参数构造正确 /
解析返回逻辑正确 / 失败时优雅降级.

不打真网络 —— 全部 mock. mootdx 端点延迟 import, 通过 monkeypatch 替换 client.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from astock_data_skill import (
    baidu_concept_blocks,
    block_trade,
    cls_telegraph,
    cninfo_announcements,
    daily_dragon_tiger,
    dedup_articles,
    dragon_tiger_board,
    eastmoney_datacenter,
    eastmoney_fund_flow_minute,
    eastmoney_global_news,
    eastmoney_reports,
    eastmoney_stock_info,
    eastmoney_stock_news,
    get_prefix,
    get_secid,
    holder_num_change,
    hsgt_realtime,
    industry_comparison,
    iwencai_search,
    load_northbound_history,
    lockup_expiry,
    margin_trading,
    normalize_ticker,
    save_northbound_snapshot,
    sina_financial_report,
    stock_fund_flow_120d,
    tencent_quote,
    ths_hot_reason,
)


# ===========================================================================
# 共用 fixture: 假的 requests.get / requests.post
# ===========================================================================


def _make_resp(payload, status_code=200, text=None):
    """模拟 requests.Response."""
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = payload
    m.text = text if text is not None else json.dumps(payload)
    m.content = (text or json.dumps(payload)).encode("utf-8")
    return m


# ===========================================================================
# _common: 归一化
# ===========================================================================


class TestNormalize:
    def test_pure_six_digit(self):
        assert normalize_ticker("688017") == "688017"

    def test_sh_prefix(self):
        assert normalize_ticker("SH688017") == "688017"
        assert normalize_ticker("sh688017") == "688017"

    def test_dot_suffix(self):
        assert normalize_ticker("688017.SH") == "688017"
        assert normalize_ticker("000001.SZ") == "000001"

    def test_bj(self):
        assert normalize_ticker("BJ832000") == "832000"

    def test_int_input(self):
        assert normalize_ticker(600519) == "600519"

    def test_pad_to_six(self):
        assert normalize_ticker("1") == "000001"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            normalize_ticker("abc123def")

    def test_get_prefix(self):
        assert get_prefix("688017") == "sh"
        assert get_prefix("600519") == "sh"
        assert get_prefix("000001") == "sz"
        assert get_prefix("300750") == "sz"
        assert get_prefix("832000") == "bj"

    def test_get_secid(self):
        assert get_secid("688017") == "1.688017"
        assert get_secid("000001") == "0.000001"


# ===========================================================================
# Helper: eastmoney_datacenter
# ===========================================================================


class TestEastmoneyDatacenter:
    @patch("astock_data_skill._common.requests.get")
    def test_returns_data_list(self, mock_get):
        mock_get.return_value = _make_resp({
            "result": {"data": [{"k": "v"}]},
        })
        result = eastmoney_datacenter("RPT_FOO")
        assert result == [{"k": "v"}]
        # 参数构造校验
        call = mock_get.call_args
        params = call.kwargs["params"]
        assert params["reportName"] == "RPT_FOO"
        assert params["pageSize"] == "50"
        assert call.kwargs["timeout"] >= 10

    @patch("astock_data_skill._common.requests.get")
    def test_empty_result_returns_empty_list(self, mock_get):
        mock_get.return_value = _make_resp({"result": None})
        assert eastmoney_datacenter("RPT_FOO") == []

    @patch("astock_data_skill._common.requests.get")
    def test_http_error_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("network down")
        assert eastmoney_datacenter("RPT_FOO") == []


# ===========================================================================
# Layer 1 — quotes
# ===========================================================================


class TestTencentQuote:
    @patch("astock_data_skill.quotes.urllib.request.urlopen")
    def test_parse_fields(self, mock_open):
        # 腾讯 GBK 响应, ~ 分隔 88 字段 (这里至少 53)
        vals = ["v_sh688017"]  # 0=key
        # 用 SKILL.md L278-302 的字段 index 设置
        vals_inner = [""] * 53
        vals_inner[1] = "绿的谐波"
        vals_inner[3] = "224.12"
        vals_inner[4] = "215.01"
        vals_inner[5] = "214.10"
        vals_inner[31] = "9.11"
        vals_inner[32] = "4.24"
        vals_inner[33] = "229.62"
        vals_inner[34] = "214.10"
        vals_inner[37] = "187040"
        vals_inner[38] = "4.55"
        vals_inner[39] = "300.45"
        vals_inner[43] = "7.22"
        vals_inner[44] = "410.88"
        vals_inner[45] = "410.88"
        vals_inner[46] = "11.51"
        vals_inner[47] = "258.01"
        vals_inner[48] = "172.01"
        vals_inner[49] = "1.20"
        vals_inner[52] = "314.76"
        inner_str = "~".join(vals_inner)
        line = f'v_sh688017="{inner_str}";'

        mock_resp = MagicMock()
        mock_resp.read.return_value = line.encode("gbk")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_open.return_value = mock_resp

        result = tencent_quote(["688017"])
        assert "688017" in result
        q = result["688017"]
        assert q["name"] == "绿的谐波"
        assert q["price"] == 224.12
        assert q["pe_ttm"] == 300.45
        assert q["pb"] == 11.51
        assert q["mcap_yi"] == 410.88
        assert q["limit_up"] == 258.01

    @patch("astock_data_skill.quotes.urllib.request.urlopen")
    def test_network_error_returns_empty(self, mock_open):
        mock_open.side_effect = Exception("DNS fail")
        assert tencent_quote(["600519"]) == {}

    def test_empty_input(self):
        assert tencent_quote([]) == {}


# ===========================================================================
# Layer 2 — research
# ===========================================================================


class TestEastmoneyReports:
    @patch("astock_data_skill.research.requests.Session")
    def test_paginated_collect(self, mock_session_cls):
        sess = MagicMock()
        # 第一页 100 条, 第二页空 → 停
        sess.get.side_effect = [
            _make_resp({"data": [{"title": "T1"}] * 3, "TotalPage": 2}),
            _make_resp({"data": []}),
        ]
        mock_session_cls.return_value = sess

        result = eastmoney_reports("688017", max_pages=3)
        assert len(result) == 3
        assert result[0]["title"] == "T1"

    @patch("astock_data_skill.research.requests.Session")
    def test_http_error_breaks(self, mock_session_cls):
        sess = MagicMock()
        sess.get.side_effect = Exception("boom")
        mock_session_cls.return_value = sess
        assert eastmoney_reports("600519") == []


class TestDedupArticles:
    def test_dedup_by_uid_keeps_max_score(self):
        articles = [
            {"uid": "a", "score": 0.5, "publish_date": "2026-01-01", "title": "low"},
            {"uid": "a", "score": 0.9, "publish_date": "2026-02-01", "title": "high"},
            {"uid": "b", "score": 0.7, "publish_date": "2026-01-15", "title": "b"},
        ]
        result = dedup_articles(articles)
        assert len(result) == 2
        a_entry = next(r for r in result if r["uid"] == "a")
        assert a_entry["title"] == "high"


class TestIwencaiSearch:
    def test_no_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("IWENCAI_API_KEY", raising=False)
        result = iwencai_search("人形机器人")
        assert result == []

    @patch("astock_data_skill.research.requests.post")
    def test_with_key_calls_api(self, mock_post, monkeypatch):
        monkeypatch.setenv("IWENCAI_API_KEY", "fake")
        mock_post.return_value = _make_resp({
            "status_code": 0,
            "data": [{"title": "t1"}],
        })
        result = iwencai_search("机器人", channel="report", size=10)
        assert result == [{"title": "t1"}]
        # X-Claw headers 必须有
        headers = mock_post.call_args.kwargs["headers"]
        assert "X-Claw-Skill-Id" in headers
        assert headers["Authorization"].startswith("Bearer ")


# ===========================================================================
# Layer 3 — signals
# ===========================================================================


class TestThsHotReason:
    @patch("astock_data_skill.signals.requests.get")
    def test_parses_and_renames(self, mock_get):
        mock_get.return_value = _make_resp({
            "errocode": 0,
            "data": [
                {"code": "688017", "name": "X", "reason": "AI+机器人",
                 "zhangfu": "10", "huanshou": "5"},
            ],
        })
        df = ths_hot_reason("2026-05-19")
        assert not df.empty
        assert "题材归因" in df.columns
        assert df.iloc[0]["题材归因"] == "AI+机器人"

    @patch("astock_data_skill.signals.requests.get")
    def test_network_fail_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("boom")
        assert ths_hot_reason().empty


class TestHsgtRealtime:
    @patch("astock_data_skill.signals.requests.get")
    def test_aligns_lists(self, mock_get):
        mock_get.return_value = _make_resp({
            "time": ["09:30", "09:31", "09:32"],
            "hgt": [1.1, 1.2],
            "sgt": [0.5, 0.6, 0.7],
        })
        df = hsgt_realtime()
        assert len(df) == 3
        assert df.iloc[0]["hgt_yi"] == 1.1
        assert df.iloc[2]["hgt_yi"] is None or df.iloc[2]["hgt_yi"] != df.iloc[2]["hgt_yi"]


class TestNorthboundCache:
    def test_save_and_load(self, tmp_path, monkeypatch):
        # 把 cache 路径重定向到 tmp 目录
        fake_cache = tmp_path / "nb.csv"
        monkeypatch.setattr(
            "astock_data_skill.signals._northbound_cache_path",
            lambda: fake_cache,
        )
        save_northbound_snapshot("2026-05-01", 100.0, 50.0)
        save_northbound_snapshot("2026-05-02", 110.0, 55.0)
        save_northbound_snapshot("2026-05-01", 105.0, 52.0)  # update

        hist = load_northbound_history(10)
        assert len(hist) == 2
        # 2026-05-01 应被更新
        d1 = hist[hist["date"] == "2026-05-01"].iloc[0]
        assert d1["hgt"] == 105.0


class TestBaiduConceptBlocks:
    @patch("astock_data_skill.signals.requests.get")
    def test_classifies(self, mock_get):
        mock_get.return_value = _make_resp({
            "ResultCode": "0",
            "Result": [
                {"type": "行业", "list": [{"name": "白酒", "increase": "1.2"}]},
                {"type": "概念", "list": [{"name": "高端消费", "increase": "0.5"}]},
                {"type": "地域", "list": [{"name": "贵州", "increase": "0.1"}]},
            ],
        })
        result = baidu_concept_blocks("600519")
        assert result["industry"][0]["name"] == "白酒"
        assert "高端消费" in result["concept_tags"]
        assert result["region"][0]["name"] == "贵州"

    @patch("astock_data_skill.signals.requests.get")
    def test_int_resultcode_zero(self, mock_get):
        # SKILL.md 踩坑: ResultCode 可能是 int(0) 也可能是 "0"
        mock_get.return_value = _make_resp({
            "ResultCode": 0,
            "Result": [{"type": "概念", "list": [{"name": "AI"}]}],
        })
        result = baidu_concept_blocks("000001")
        assert "AI" in result["concept_tags"]


class TestEastmoneyFundFlowMinute:
    @patch("astock_data_skill.signals.requests.get")
    def test_parse_klines(self, mock_get):
        mock_get.return_value = _make_resp({
            "data": {"klines": [
                "09:30,1000,200,300,400,500,600",
                "09:31,1100,210,310,410,510,610",
            ]},
        })
        result = eastmoney_fund_flow_minute("000858")
        assert len(result) == 2
        assert result[0]["time"] == "09:30"
        assert result[0]["main_net"] == 1000.0

    @patch("astock_data_skill.signals.requests.get")
    def test_secid_for_sh(self, mock_get):
        mock_get.return_value = _make_resp({"data": {"klines": []}})
        eastmoney_fund_flow_minute("688017")
        params = mock_get.call_args.kwargs["params"]
        assert params["secid"] == "1.688017"


class TestDragonTigerBoard:
    @patch("astock_data_skill._common.requests.get")
    def test_aggregates(self, mock_get):
        # records / buy seats / sell seats 三次调用
        mock_get.side_effect = [
            _make_resp({"result": {"data": [{
                "TRADE_DATE": "2026-05-17 00:00:00",
                "EXPLANATION": "日涨幅偏离值7%",
                "BILLBOARD_NET_AMT": 5_000_000,
                "TURNOVERRATE": 12.34,
            }]}}),
            _make_resp({"result": {"data": [{
                "OPERATEDEPT_NAME": "机构专用",
                "OPERATEDEPT_CODE": "0",
                "BUY": 3_000_000,
                "SELL": 0,
                "NET": 3_000_000,
            }]}}),
            _make_resp({"result": {"data": []}}),
        ]
        result = dragon_tiger_board("002475", "2026-05-17", look_back=30)
        assert len(result["records"]) == 1
        assert result["records"][0]["net_buy"] == 500.0  # 5_000_000/10000
        assert len(result["seats"]["buy"]) == 1
        assert result["institution"]["buy_amt"] == 300.0


class TestDailyDragonTiger:
    @patch("astock_data_skill._common.requests.get")
    def test_aggregates_market(self, mock_get):
        mock_get.return_value = _make_resp({"result": {"data": [
            {
                "TRADE_DATE": "2026-05-17 00:00:00",
                "SECURITY_CODE": "600519",
                "SECURITY_NAME_ABBR": "贵州茅台",
                "EXPLANATION": "无价格涨跌幅限制证券",
                "CLOSE_PRICE": 1500.0,
                "CHANGE_RATE": 2.5,
                "BILLBOARD_NET_AMT": 100_000_000,
                "BILLBOARD_BUY_AMT": 150_000_000,
                "BILLBOARD_SELL_AMT": 50_000_000,
                "TURNOVERRATE": 1.5,
            },
        ]}})
        result = daily_dragon_tiger("2026-05-17", min_net_buy=5000)
        assert result["total_records"] == 1
        assert result["stocks"][0]["code"] == "600519"


class TestLockupExpiry:
    @patch("astock_data_skill._common.requests.get")
    def test_history_and_upcoming(self, mock_get):
        mock_get.side_effect = [
            _make_resp({"result": {"data": [{
                "FREE_DATE": "2025-01-15 00:00:00",
                "LIMITED_STOCK_TYPE": "首发原股东",
                "FREE_SHARES_NUM": 1e8,
                "FREE_RATIO": 5.0,
            }]}}),
            _make_resp({"result": {"data": []}}),
        ]
        result = lockup_expiry("002475", "2026-05-17", forward_days=90)
        assert result["history"][0]["type"] == "首发原股东"
        assert result["upcoming"] == []


class TestIndustryComparison:
    @patch("astock_data_skill.signals.requests.get")
    def test_top_bottom(self, mock_get):
        diff = []
        for i in range(50):
            diff.append({
                "f14": f"行业{i}", "f3": i, "f12": f"BK{i:04d}",
                "f104": 10, "f105": 5, "f140": "领涨", "f136": 5,
            })
        mock_get.return_value = _make_resp({"data": {"diff": diff}})
        result = industry_comparison(top_n=5)
        assert result["total"] == 50
        assert len(result["top"]) == 5
        assert len(result["bottom"]) == 5
        assert result["top"][0]["rank"] == 1


# ===========================================================================
# Layer 4 — flows
# ===========================================================================


class TestMarginTrading:
    @patch("astock_data_skill._common.requests.get")
    def test_returns_rows(self, mock_get):
        mock_get.return_value = _make_resp({"result": {"data": [{
            "DATE": "2026-05-17 00:00:00",
            "RZYE": 1e10, "RZMRE": 2e8, "RZCHE": 1e8,
            "RQYE": 1e8, "RQMCL": 1e6, "RQCHL": 5e5,
            "RZRQYE": 1.01e10,
        }]}})
        result = margin_trading("600519")
        assert len(result) == 1
        assert result[0]["rzye"] == 1e10


class TestBlockTrade:
    @patch("astock_data_skill._common.requests.get")
    def test_premium_calc(self, mock_get):
        mock_get.return_value = _make_resp({"result": {"data": [{
            "TRADE_DATE": "2026-05-17 00:00:00",
            "CLOSE_PRICE": 100.0, "DEAL_PRICE": 90.0,
            "DEAL_VOLUME": 1e7, "DEAL_AMT": 9e8,
            "BUYER_NAME": "A", "SELLER_NAME": "B",
        }]}})
        result = block_trade("600519")
        assert len(result) == 1
        assert result[0]["premium_pct"] == -10.0


class TestHolderNumChange:
    @patch("astock_data_skill._common.requests.get")
    def test_returns_rows(self, mock_get):
        mock_get.return_value = _make_resp({"result": {"data": [{
            "END_DATE": "2026-03-31 00:00:00",
            "HOLDER_NUM": 100000, "HOLDER_NUM_CHANGE": -1000,
            "HOLDER_NUM_RATIO": -1.0, "AVG_FREE_SHARES": 5000,
        }]}})
        result = holder_num_change("600519")
        assert result[0]["holder_num"] == 100000


class TestStockFundFlow120d:
    @patch("astock_data_skill.flows.requests.get")
    def test_parse_klines(self, mock_get):
        mock_get.return_value = _make_resp({"data": {"klines": [
            "2026-05-17,1e6,2e5,3e5,4e5,5e5,6e5,7e5",
            "2026-05-18,1.1e6,2.1e5,3.1e5,4.1e5,5.1e5,6.1e5,7.1e5",
        ]}})
        result = stock_fund_flow_120d("600519")
        assert len(result) == 2
        assert result[0]["main_net"] == 1e6

    @patch("astock_data_skill.flows.requests.get")
    def test_dash_becomes_zero(self, mock_get):
        mock_get.return_value = _make_resp({"data": {"klines": [
            "2026-05-17,-,-,-,-,-,-,-",
        ]}})
        result = stock_fund_flow_120d("600519")
        assert result[0]["main_net"] == 0.0


# ===========================================================================
# Layer 5 — news
# ===========================================================================


class TestEastmoneyStockNews:
    @patch("astock_data_skill.news.requests.get")
    def test_parse_jsonp(self, mock_get):
        payload = {
            "result": {"cmsArticleWebOld": {"list": [
                {
                    "title": "<em>茅台</em>大涨", "content": "正文<br/>内容",
                    "date": "2026-05-17 10:00", "mediaName": "财经网",
                    "url": "http://example.com/a1",
                },
            ]}},
        }
        jsonp = f"jQuery_news({json.dumps(payload)})"
        mock_get.return_value = _make_resp(payload, text=jsonp)
        result = eastmoney_stock_news("600519")
        assert len(result) == 1
        assert result[0]["title"] == "茅台大涨"  # HTML 标签被剥
        assert result[0]["source"] == "财经网"


class TestClsTelegraph:
    @patch("astock_data_skill.news.requests.get")
    def test_returns_rows(self, mock_get):
        mock_get.return_value = _make_resp({"data": {"roll_data": [
            {"title": "T1", "content": "C1", "ctime": 1716000000},
        ]}})
        result = cls_telegraph()
        assert len(result) == 1
        assert result[0]["title"] == "T1"


class TestEastmoneyGlobalNews:
    @patch("astock_data_skill.news.requests.get")
    def test_returns_rows(self, mock_get):
        mock_get.return_value = _make_resp({"data": {"fastNewsList": [
            {"title": "Hot", "summary": "Sum", "showTime": "10:00"},
        ]}})
        result = eastmoney_global_news()
        assert len(result) == 1
        assert result[0]["title"] == "Hot"


# ===========================================================================
# Layer 6 — fundamentals
# ===========================================================================


class TestEastmoneyStockInfo:
    @patch("astock_data_skill.fundamentals.requests.get")
    def test_returns_fields(self, mock_get):
        mock_get.return_value = _make_resp({"data": {
            "f57": "600519", "f58": "贵州茅台", "f127": "白酒Ⅲ",
            "f84": 1.26e9, "f85": 1.26e9,
            "f116": 1.5e12, "f117": 1.5e12,
            "f189": "20010827", "f43": 1500.0,
        }})
        info = eastmoney_stock_info("600519")
        assert info["name"] == "贵州茅台"
        assert info["industry"] == "白酒Ⅲ"
        assert info["mcap"] == 1.5e12

    @patch("astock_data_skill.fundamentals.requests.get")
    def test_empty_data(self, mock_get):
        mock_get.return_value = _make_resp({"data": None})
        assert eastmoney_stock_info("600519") == {}


class TestSinaFinancialReport:
    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            sina_financial_report("600519", report_type="xxx")

    @patch("astock_data_skill.fundamentals.requests.get")
    def test_returns_rows(self, mock_get):
        mock_get.return_value = _make_resp({"result": {"data": {"lrb": [
            {"报告日": "2026-03-31", "净利润": 1e8},
        ]}}})
        result = sina_financial_report("600519", "lrb")
        assert len(result) == 1
        assert result[0]["净利润"] == 1e8


# ===========================================================================
# Layer 7 — announcements
# ===========================================================================


class TestCninfoAnnouncements:
    @patch("astock_data_skill.announcements.requests.post")
    def test_parse_announcements(self, mock_post):
        mock_post.return_value = _make_resp({"announcements": [
            {
                "announcementTitle": "年报", "announcementTypeName": "定期报告",
                "announcementTime": "2026-04-30", "announcementId": "ANN001",
            },
        ]})
        result = cninfo_announcements("600519")
        assert len(result) == 1
        assert result[0]["title"] == "年报"
        assert "annoId=ANN001" in result[0]["url"]

    @patch("astock_data_skill.announcements.requests.post")
    def test_orgid_prefix_for_sh(self, mock_post):
        mock_post.return_value = _make_resp({"announcements": []})
        cninfo_announcements("600519")
        payload = mock_post.call_args.kwargs["data"]
        assert "gssh0600519" in payload["stock"]

    @patch("astock_data_skill.announcements.requests.post")
    def test_orgid_prefix_for_sz(self, mock_post):
        mock_post.return_value = _make_resp({"announcements": []})
        cninfo_announcements("000001")
        payload = mock_post.call_args.kwargs["data"]
        assert "gssz0000001" in payload["stock"]
