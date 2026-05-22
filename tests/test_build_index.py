"""Unit tests for scripts/build_index.py

build_index.py 在 2026-05-22 随项目「涨跌预测 → 价值选股」改造一并重写：
- 不再读 predictions_<date>.json 里的 AUC / R²（价值版 JSON 没有这些字段）
- 改为从 artifacts/quarterly_backtest/results_<date>.json 读回测指标
- 归档表简化为「日期 + 方法标签 + 查看」，按 METHOD_SWITCH_DATE 区分新旧方法
本测试同步重写，覆盖这套新逻辑。
"""

import importlib.util
import json
from pathlib import Path

import pytest

# Load build_index as a module without installing it
_SCRIPT = Path(__file__).parent.parent / "scripts" / "build_index.py"
spec = importlib.util.spec_from_file_location("build_index", _SCRIPT)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)


@pytest.fixture()
def fake_project(tmp_path, monkeypatch):
    """Set up a minimal fake project tree and redirect build_index paths."""
    docs_reports = tmp_path / "docs" / "reports"
    docs_reports.mkdir(parents=True)
    backtest = tmp_path / "artifacts" / "quarterly_backtest"
    backtest.mkdir(parents=True)

    monkeypatch.setattr(_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(_mod, "DOCS_REPORTS", docs_reports)
    monkeypatch.setattr(_mod, "BACKTEST_DIR", backtest)
    monkeypatch.setattr(_mod, "INDEX_HTML", tmp_path / "docs" / "index.html")

    return tmp_path


def _write_html(docs_reports: Path, date_str: str) -> None:
    (docs_reports / f"daily_report_{date_str}.html").write_text(
        f"<html><body>report {date_str}</body></html>", encoding="utf-8"
    )


def _write_backtest(
    backtest: Path,
    date_str: str,
    total_return=0.2073,
    benchmark=-0.0796,
    excess=0.0625,
    ir=0.46,
) -> None:
    """写一份回测 artifact —— 结构对齐真实 results_<date>.json 的 metrics 段。"""
    payload = {
        "meta": {"date": date_str},
        "metrics": {
            "total_return": total_return,
            "benchmark_total_return": benchmark,
            "excess_return_annualized": excess,
            "information_ratio": ir,
            "max_drawdown": -0.2703,
            "start_date": "2022-01-04",
            "end_date": "2026-04-01",
        },
    }
    (backtest / f"results_{date_str}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_index_generated_with_reports(fake_project):
    """index.html 生成成功，含报告日期、价值选股主题、回测卡片数字。"""
    docs_reports = fake_project / "docs" / "reports"
    backtest = fake_project / "artifacts" / "quarterly_backtest"

    _write_html(docs_reports, "2026-05-22")
    _write_backtest(backtest, "2026-05-22")

    _mod.build_index()

    index = (fake_project / "docs" / "index.html").read_text(encoding="utf-8")
    assert "2026-05-22" in index
    assert "价值选股" in index
    # 回测卡片：策略累计收益 / 沪深300 同期 / 年化超额
    assert "+20.73%" in index
    assert "-7.96%" in index
    assert "+6.25%" in index
    assert "非投资建议" in index
    assert "cdn" not in index.lower()
    # 旧的「涨跌预测」指标不应再出现
    assert "方向 AUC" not in index
    assert "收益 R²" not in index


def test_index_no_backtest_artifact(fake_project):
    """没有回测 artifact 时，首页仍能生成，只是不显示回测卡片。"""
    docs_reports = fake_project / "docs" / "reports"
    _write_html(docs_reports, "2026-05-22")

    _mod.build_index()

    index = (fake_project / "docs" / "index.html").read_text(encoding="utf-8")
    assert "2026-05-22" in index
    assert "价值选股" in index
    # 回测卡片缺数据时整块不渲染 —— 查 div（class 名在 <style> 里始终存在，不能用它判断）
    assert '<div class="backtest-strip">' not in index
    assert "策略累计收益" not in index


def test_index_empty_reports(fake_project):
    """docs/reports/ 为空时 index.html 仍能生成。"""
    _mod.build_index()

    index = (fake_project / "docs" / "index.html").read_text(encoding="utf-8")
    assert "A股" in index
    assert "非投资建议" in index


def test_index_multiple_reports_sorted(fake_project):
    """报告按日期降序排列，最新的在最前。"""
    docs_reports = fake_project / "docs" / "reports"
    for d in ["2026-05-17", "2026-05-19", "2026-05-22"]:
        _write_html(docs_reports, d)

    _mod.build_index()

    index = (fake_project / "docs" / "index.html").read_text(encoding="utf-8")
    pos_22 = index.index("2026-05-22")
    pos_19 = index.index("2026-05-19")
    pos_17 = index.index("2026-05-17")
    assert pos_22 < pos_19 < pos_17


def test_index_method_tags(fake_project):
    """切换日(含)起的报告标「价值选股」，更早的标「旧·涨跌预测」。"""
    docs_reports = fake_project / "docs" / "reports"
    # 一份切换日当天，一份切换日之前
    _write_html(docs_reports, _mod.METHOD_SWITCH_DATE)
    _write_html(docs_reports, "2026-05-16")

    _mod.build_index()

    index = (fake_project / "docs" / "index.html").read_text(encoding="utf-8")
    assert "价值选股" in index
    assert "旧·涨跌预测" in index


def test_index_meta_tags(fake_project):
    """生成的 index.html 含必要的 charset / viewport meta。"""
    _mod.build_index()

    index = (fake_project / "docs" / "index.html").read_text(encoding="utf-8")
    assert 'charset="UTF-8"' in index
    assert 'name="viewport"' in index
