"""P13 — launchd plist 模板 + wrapper shell 脚本的静态验证测试."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PLIST_TEMPLATE = PROJECT_ROOT / "scripts" / "com.astock.daily.plist.template"
WRAPPER_SH = PROJECT_ROOT / "scripts" / "daily_predict_wrapper.sh"


class TestPlistTemplate:
    def test_plist_file_exists(self):
        assert PLIST_TEMPLATE.exists(), f"plist 模板不存在：{PLIST_TEMPLATE}"

    def test_plist_contains_start_calendar_interval(self):
        content = PLIST_TEMPLATE.read_text()
        assert "StartCalendarInterval" in content

    def test_plist_schedule_is_1630(self):
        content = PLIST_TEMPLATE.read_text()
        assert "<integer>16</integer>" in content, "Hour 应为 16"
        assert "<integer>30</integer>" in content, "Minute 应为 30"

    def test_plist_contains_program_arguments(self):
        content = PLIST_TEMPLATE.read_text()
        assert "ProgramArguments" in content

    def test_plist_contains_standard_out_path(self):
        content = PLIST_TEMPLATE.read_text()
        assert "StandardOutPath" in content

    def test_plist_contains_standard_error_path(self):
        content = PLIST_TEMPLATE.read_text()
        assert "StandardErrorPath" in content

    def test_plist_run_at_load_is_false(self):
        content = PLIST_TEMPLATE.read_text()
        # RunAtLoad key must be present and followed by <false/>
        assert "RunAtLoad" in content
        idx = content.index("RunAtLoad")
        snippet = content[idx : idx + 100]
        assert "<false/>" in snippet, "RunAtLoad 应为 false（开机不自动跑）"

    def test_plist_uses_project_root_placeholder(self):
        content = PLIST_TEMPLATE.read_text()
        assert "{project_root}" in content, "plist 模板应含 {project_root} 占位符，不应 hardcode 路径"

    def test_plist_label_is_com_astock_daily(self):
        content = PLIST_TEMPLATE.read_text()
        assert "com.astock.daily" in content


class TestWrapperShell:
    def test_wrapper_file_exists(self):
        assert WRAPPER_SH.exists(), f"wrapper 脚本不存在：{WRAPPER_SH}"

    def test_wrapper_syntax_ok(self):
        result = subprocess.run(
            ["bash", "-n", str(WRAPPER_SH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"bash -n 语法检查失败：{result.stderr}"

    def test_wrapper_contains_osascript_notification(self):
        content = WRAPPER_SH.read_text()
        assert "osascript" in content, "wrapper 应含 osascript 通知调用"

    def test_wrapper_contains_success_and_failure_notify(self):
        content = WRAPPER_SH.read_text()
        assert "今日预测报告已生成" in content
        assert "预测脚本失败" in content

    def test_wrapper_exits_zero_on_failure(self):
        content = WRAPPER_SH.read_text()
        assert "exit 0" in content, "wrapper 应始终 exit 0，不阻塞 launchd 下次调度"

    def test_wrapper_writes_error_log(self):
        content = WRAPPER_SH.read_text()
        assert "ERROR_LOG" in content or "error" in content.lower()

    def test_wrapper_opens_html_report(self):
        content = WRAPPER_SH.read_text()
        assert "open" in content and ".html" in content
