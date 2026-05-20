#!/usr/bin/env python3
"""生成 docs/index.html 报告归档列表首页。无参数，从 PROJECT_ROOT 运行。"""

import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_REPORTS = PROJECT_ROOT / "docs" / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "daily_reports"
INDEX_HTML = PROJECT_ROOT / "docs" / "index.html"

DATE_RE = re.compile(r"daily_report_(\d{4}-\d{2}-\d{2})\.html$")


def load_metrics(report_date: str) -> dict:
    json_path = ARTIFACTS_DIR / f"predictions_{report_date}.json"
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", {})
        direction = results.get("direction", {}).get("metrics", results.get("direction", {}))
        return_ = results.get("return_", {}).get("metrics", results.get("return_", {}))
        return {
            "auc": direction.get("auc"),
            "r2": return_.get("r2"),
            "n_predictions": direction.get("n_predictions"),
        }
    except Exception:
        return {}


def fmt_metric(val, fmt=".4f") -> str:
    if val is None:
        return "N/A"
    return format(val, fmt)


def build_row(report_date: str, metrics: dict, is_latest: bool) -> str:
    auc = fmt_metric(metrics.get("auc"))
    r2 = fmt_metric(metrics.get("r2"))
    n = metrics.get("n_predictions")
    n_str = str(n) if n is not None else "N/A"
    link = f"reports/daily_report_{report_date}.html"
    row_class = "latest-row" if is_latest else "hist-row"
    return (
        f'<tr class="{row_class}">'
        f'<td><a href="{link}">{report_date}</a></td>'
        f"<td>{auc}</td>"
        f"<td>{r2}</td>"
        f"<td>{n_str}</td>"
        f'<td><a href="{link}" class="btn">查看 →</a></td>'
        f"</tr>\n"
    )


def build_index() -> None:
    DOCS_REPORTS.mkdir(parents=True, exist_ok=True)

    report_files = sorted(
        [f for f in DOCS_REPORTS.glob("daily_report_*.html") if DATE_RE.search(f.name)],
        key=lambda f: f.name,
        reverse=True,
    )

    today_str = date.today().isoformat()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    if report_files:
        latest_date = DATE_RE.search(report_files[0].name).group(1)
        latest_exists = True
    else:
        latest_date = today_str
        latest_exists = False

    latest_metrics = load_metrics(latest_date) if latest_exists else {}
    latest_auc = fmt_metric(latest_metrics.get("auc"))
    latest_r2 = fmt_metric(latest_metrics.get("r2"))
    latest_n = latest_metrics.get("n_predictions")
    latest_n_str = str(latest_n) if latest_n is not None else "N/A"

    if latest_exists:
        latest_status = f'<a href="reports/daily_report_{latest_date}.html" class="btn-primary">点击查看 →</a>'
        latest_badge = '<span class="badge ok">已生成</span>'
    else:
        latest_status = '<span class="badge warn">暂无报告</span>'
        latest_badge = '<span class="badge warn">暂无</span>'

    rows_html = ""
    for i, f in enumerate(report_files[:30]):
        d = DATE_RE.search(f.name).group(1)
        m = load_metrics(d)
        rows_html += build_row(d, m, i == 0)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A股 每日预测报告 · 归档</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, "PingFang SC", "Microsoft YaHei", "Hiragino Sans GB", sans-serif;
    background: #f5f5f5;
    color: #333;
    min-height: 100vh;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #fff;
    padding: 2rem 1rem;
    text-align: center;
  }}
  .header h1 {{ font-size: 1.6rem; font-weight: 700; letter-spacing: 0.05em; }}
  .header .subtitle {{ margin-top: 0.4rem; font-size: 0.9rem; opacity: 0.7; }}
  .container {{ max-width: 860px; margin: 0 auto; padding: 1.5rem 1rem; }}
  .card {{
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }}
  .card-title {{ font-size: 1rem; font-weight: 600; color: #555; margin-bottom: 1rem; }}
  .latest-info {{ display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }}
  .latest-date {{ font-size: 2rem; font-weight: 700; color: #1a1a2e; }}
  .badge {{
    display: inline-block; padding: 0.25rem 0.6rem; border-radius: 4px;
    font-size: 0.8rem; font-weight: 600;
  }}
  .badge.ok {{ background: #d4edda; color: #155724; }}
  .badge.warn {{ background: #fff3cd; color: #856404; }}
  .metrics-row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin-top: 0.8rem; }}
  .metric {{ font-size: 0.85rem; color: #666; }}
  .metric span {{ font-weight: 600; color: #333; }}
  .btn-primary {{
    display: inline-block; background: #1a1a2e; color: #fff;
    padding: 0.5rem 1.2rem; border-radius: 6px; text-decoration: none;
    font-size: 0.9rem; font-weight: 500; margin-top: 0.8rem;
  }}
  .btn-primary:hover {{ background: #2a2a4e; }}
  .btn {{
    color: #1a1a2e; text-decoration: none; font-size: 0.85rem; font-weight: 500;
  }}
  .btn:hover {{ text-decoration: underline; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ background: #f0f0f0; text-align: left; padding: 0.6rem 0.8rem; font-size: 0.8rem; color: #666; }}
  td {{ padding: 0.55rem 0.8rem; border-bottom: 1px solid #f0f0f0; }}
  tr.latest-row td {{ background: #f0f6ff; font-weight: 600; }}
  tr:hover td {{ background: #fafafa; }}
  .footer {{
    text-align: center; padding: 1.5rem 1rem 2rem;
    font-size: 0.75rem; color: #999; line-height: 1.6;
  }}
  .disclaimer {{
    background: #fff8e1; border-left: 3px solid #f9a825;
    padding: 0.8rem 1rem; border-radius: 0 6px 6px 0;
    font-size: 0.8rem; color: #5d4037; line-height: 1.6;
  }}
  @media (max-width: 600px) {{
    .latest-date {{ font-size: 1.4rem; }}
    table {{ font-size: 0.8rem; }}
    th, td {{ padding: 0.45rem 0.5rem; }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>A股 每日预测报告 · 归档</h1>
  <div class="subtitle">量化学习研究项目 · 更新于 {generated_at}</div>
</div>
<div class="container">

  <div class="card">
    <div class="card-title">今日报告（{latest_date}）</div>
    <div class="latest-info">
      <div class="latest-date">{latest_date}</div>
      {latest_badge}
    </div>
    <div class="metrics-row">
      <div class="metric">方向 AUC <span>{latest_auc}</span></div>
      <div class="metric">收益 R² <span>{latest_r2}</span></div>
      <div class="metric">预测标的 <span>{latest_n_str}</span></div>
    </div>
    <div>{latest_status}</div>
  </div>

  <div class="card">
    <div class="card-title">历史归档（最近 30 条）</div>
    <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>日期</th>
          <th>方向 AUC</th>
          <th>收益 R²</th>
          <th>标的数</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
{rows_html}      </tbody>
    </table>
    </div>
  </div>

  <div class="disclaimer">
    ⚠️ 本站为量化学习 / 研究项目，所有预测结果仅供学习参考，<strong>不构成任何投资建议</strong>。
    市场有风险，投资须谨慎。模型预测存在误差，请勿据此进行实盘操作。
  </div>

</div>
<div class="footer">
  A股量化预测系统 · Stage 4 · 数据来源：AkShare &nbsp;|&nbsp; 非投资建议
</div>
</body>
</html>
"""
    INDEX_HTML.write_text(html, encoding="utf-8")
    print(f"[build_index] 已写入 {INDEX_HTML}（{len(report_files)} 条归档）")


if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        print(f"[build_index] 错误：{e}", file=sys.stderr)
        sys.exit(1)
