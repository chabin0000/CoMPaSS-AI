"""집 데스크톱에서 실행하는 연구 준비 상태 점검."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .jsonl import read_jsonl
from .leakage import assert_private_final_is_safe
from .pilot import assess_pilot_questions, create_pilot_manifest, validate_pilot_questions, write_pilot_manifest


def pre_dgx_check(root: Path, run_tests: bool = True) -> dict:
    """프로그램 오류(error)와 아직 작성하지 않은 연구 입력(not_ready)을 구분한다."""
    result = {"status": "ready", "errors": [], "not_ready_reasons": [], "warnings": [], "tests": None, "pilot": None, "manifest": None}
    required = [root / "docs" / "dgx_runbook.md", root / "evaluation" / "private_final" / "README.md"]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        result["errors"].append({"type": "required_file_missing", "files": missing})
    try:
        assert_private_final_is_safe(root / "evaluation" / "private_final")
    except Exception as error:
        result["errors"].append({"type": "private_final_protection", "message": str(error)})
    pilot_file = root / "data" / "questions" / "pilot" / "pilot_questions.jsonl"
    if not pilot_file.exists() or not pilot_file.read_text(encoding="utf-8").strip():
        result["not_ready_reasons"].append("actual pilot dataset missing")
    else:
        try:
            records = validate_pilot_questions(read_jsonl(pilot_file))
            report = assess_pilot_questions(records, private_ids=[])
            result["pilot"] = report
            result["warnings"].extend(report["warnings"])
            result["errors"].extend(report["errors"])
            if not report["errors"]:
                manifest = create_pilot_manifest(pilot_file, records)
                write_pilot_manifest(pilot_file.parent / "pilot_manifest.json", manifest)
                result["manifest"] = manifest
        except Exception as error:
            result["errors"].append({"type": "pilot_validation", "message": str(error)})
    if run_tests:
        completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, text=True, capture_output=True)
        result["tests"] = {"returncode": completed.returncode, "output": completed.stdout.strip()}
        if completed.returncode:
            result["errors"].append({"type": "pytest_failed", "message": completed.stdout[-1000:]})
    if result["errors"]:
        result["status"] = "error"
    elif result["not_ready_reasons"]:
        result["status"] = "not_ready"
    return result
