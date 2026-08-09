"""재현 가능한 실행 로그 생성."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


@dataclass
class RunLogger:
    path: Path

    def event(self, event_type: str, payload: dict) -> None:
        row = {"timestamp": datetime.now(UTC).isoformat(), "event": event_type, **payload}
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_metrics(self, metrics: dict) -> None:
        (self.path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def create_run_logger(log_root: Path, config: dict, command: list[str]) -> RunLogger:
    run_path = log_root / f"{datetime.now().strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    run_path.mkdir(parents=True)
    (run_path / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    environment = {"timestamp": datetime.now(UTC).isoformat(), "python": platform.python_version(), "os": platform.platform(), "architecture": platform.machine(), "git_commit": _git_hash(), "command": command}
    (run_path / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_path / "events.jsonl").touch()
    return RunLogger(run_path)
