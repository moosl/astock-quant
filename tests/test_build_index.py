"""Unit tests for scripts/build_index.py"""

import importlib.util
import json
import sys
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
    artifacts = tmp_path / "artifacts" / "daily_reports"
    artifacts.mkdir(parents=True)

    monkeypatch.setattr(_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(_mod, "DOCS_REPORTS", docs_reports)
    monkeypatch.setattr(_mod, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(_mod, "INDEX_HTML", tmp_path / "docs" / "index.html")

    return tmp_path


def _write_html(docs_reports: Path, date_str: str) -> None:
    (docs_reports / f"daily_report_{date_str}.html").write_text(
        f"<html><body>report {date_str}</body></html>", encoding="utf-8"
    )


def _write_json(artifacts: Path, date_str: str, auc=0.51, r2=0.03, n=300) -> None:
    payload = {
        "report_date": date_str,
        "results": {
            "direction": {"auc": auc, "n_predictions": n},
            "return_": {"r2": r2},
        },
    }
    (artifacts / f"predictions_{date_str}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_index_generated_with_reports(fake_project):
    """index.html is created and contains expected date and metrics."""
    docs_reports = fake_project / "docs" / "reports"
    artifacts = fake_project / "artifacts" / "daily_reports"

    _write_html(docs_reports, "2026-05-19")
    _write_json(artifacts, "2026-05-19", auc=0.4898, r2=-0.005, n=300)

    _mod.build_index()

    index = (fake_project / "docs" / "index.html").read_text(encoding="utf-8")
    assert "2026-05-19" in index
    assert "0.4898" in index
    assert "-0.0050" in index
    assert "300" in index
    assert "非投资建议" in index
    assert "cdn" not in index.lower()


def test_index_generated_no_json(fake_project):
    """index.html is generated even when no JSON metrics are available."""
    docs_reports = fake_project / "docs" / "reports"
    _write_html(docs_reports, "2026-05-18")

    _mod.build_index()

    index = (fake_project / "docs" / "index.html").read_text(encoding="utf-8")
    assert "2026-05-18" in index
    assert "N/A" in index


def test_index_empty_reports(fake_project):
    """index.html is generated even when docs/reports/ is empty."""
    _mod.build_index()

    index = (fake_project / "docs" / "index.html").read_text(encoding="utf-8")
    assert "A股" in index
    assert "非投资建议" in index


def test_index_multiple_reports_sorted(fake_project):
    """Reports appear in descending date order; latest appears first."""
    docs_reports = fake_project / "docs" / "reports"
    for d in ["2026-05-17", "2026-05-18", "2026-05-19"]:
        _write_html(docs_reports, d)

    _mod.build_index()

    index = (fake_project / "docs" / "index.html").read_text(encoding="utf-8")
    pos_19 = index.index("2026-05-19")
    pos_18 = index.index("2026-05-18")
    pos_17 = index.index("2026-05-17")
    assert pos_19 < pos_18 < pos_17


def test_index_meta_tags(fake_project):
    """Generated index.html has required meta charset and viewport tags."""
    _mod.build_index()

    index = (fake_project / "docs" / "index.html").read_text(encoding="utf-8")
    assert 'charset="UTF-8"' in index
    assert 'name="viewport"' in index
