from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from astock_quant.llm.client import LLMClientError
from astock_quant.llm.codex_cli import CodexCLIClient


class Decision(BaseModel):
    action: str
    confidence: float
    note: str | None = None


def test_chat_uses_isolated_codex_exec_command():
    seen_workdir: Path | None = None

    def run(cmd, **kwargs):
        nonlocal seen_workdir
        seen_workdir = Path(cmd[cmd.index("--cd") + 1])
        assert seen_workdir.is_dir()
        assert seen_workdir != Path.cwd()
        output = Path(cmd[cmd.index("--output-last-message") + 1])
        output.write_text("answer", encoding="utf-8")
        assert kwargs["env"]["CODEX_HOME"] == "/tmp/test-codex-home"
        return CompletedProcess(cmd, 0, "", "")

    client = CodexCLIClient(model="gpt-test", cli_home="/tmp/test-codex-home")
    with (
        patch("astock_quant.llm.codex_cli.shutil.which", return_value="/bin/codex"),
        patch(
            "astock_quant.llm.codex_cli._disabled_skill_config",
            return_value='skills.config=[{path="/tmp/SKILL.md",enabled=false}]',
        ),
        patch("astock_quant.llm.codex_cli.subprocess.run", side_effect=run) as mock_run,
    ):
        response = client.chat([{"role": "user", "content": "hello"}])

    cmd = mock_run.call_args.args[0]
    assert response.content == "answer"
    assert seen_workdir is not None and not seen_workdir.exists()
    assert ["--sandbox", "read-only"] == cmd[cmd.index("--sandbox") : cmd.index("--sandbox") + 2]
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--ignore-rules" in cmd
    assert "project_doc_max_bytes=0" in cmd
    assert any(value.startswith("skills.config=[") for value in cmd)
    assert ["--model", "gpt-test"] == cmd[cmd.index("--model") : cmd.index("--model") + 2]


def test_default_model_uses_codex_cli_default():
    client = CodexCLIClient()
    assert client.model == "default"


def test_chat_json_uses_output_schema():
    def run(cmd, **kwargs):
        schema_path = Path(cmd[cmd.index("--output-schema") + 1])
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"action", "confidence", "note"}
        output_path.write_text(
            '{"action":"hold","confidence":0.8,"note":null}',
            encoding="utf-8",
        )
        return CompletedProcess(cmd, 0, "", "")

    client = CodexCLIClient()
    with (
        patch("astock_quant.llm.codex_cli.shutil.which", return_value="/bin/codex"),
        patch("astock_quant.llm.codex_cli.subprocess.run", side_effect=run),
    ):
        result = client.chat_json(
            [{"role": "user", "content": "decide"}],
            Decision,
        )

    assert result == Decision(action="hold", confidence=0.8, note=None)


def test_missing_codex_cli_fails_loudly():
    client = CodexCLIClient()
    with patch("astock_quant.llm.codex_cli.shutil.which", return_value=None):
        with pytest.raises(LLMClientError, match="找不到 codex CLI"):
            client.chat([{"role": "user", "content": "hello"}])
