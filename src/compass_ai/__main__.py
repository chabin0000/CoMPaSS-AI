"""`python -m compass_ai` 명령행 인터페이스."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .jsonl import read_jsonl
from .leakage import assert_no_evaluation_leakage, assert_private_final_is_safe
from .pipeline import run_mock_pipeline
from .schemas import validate_questions, validate_responses
from .system_check import system_check
from .teacher import generate_responses, merge_responses
from .verification import verify_responses
from .conditions import build_conditions
from .jsonl import write_jsonl
from .pilot import assess_pilot_questions, create_pilot_manifest, validate_pilot_questions, write_pilot_manifest
from .pre_dgx import pre_dgx_check
from .system_check import dgx_readiness


def _questions(root: Path) -> list[dict]:
    path = root / "data" / "questions" / "mock_questions.jsonl"
    if not path.exists():
        path = Path(__file__).resolve().parents[2] / "data" / "questions" / "mock_questions.jsonl"
    return validate_questions(read_jsonl(path))


def _raw(root: Path) -> list[dict]:
    paths = [root / "data" / "raw" / "gpt_oss" / "mock_teacher_a.jsonl", root / "data" / "raw" / "glm" / "mock_teacher_b.jsonl"]
    return validate_responses([row for path in paths if path.exists() for row in read_jsonl(path)])


def _pilot_file(root: Path) -> Path:
    return root / "data" / "questions" / "pilot" / "pilot_questions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="CoMPaSS 연구 파이프라인")
    parser.add_argument("command", choices=["system-check", "dgx-readiness", "pilot-check", "pilot-manifest", "pre-dgx-check", "validate", "generate", "merge", "conflicts", "verify", "build-conditions", "mock-e2e", "leakage-check", "summary"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--file")
    args = parser.parse_args()
    root = Path(args.root)
    if args.command == "system-check":
        print(json.dumps(system_check(root), ensure_ascii=False, indent=2))
    elif args.command == "dgx-readiness":
        print(json.dumps(dgx_readiness(system_check(root)), ensure_ascii=False, indent=2))
    elif args.command in {"pilot-check", "pilot-manifest"}:
        path = Path(args.file) if args.file else _pilot_file(root)
        if not path.exists():
            print("not ready: actual pilot dataset missing")
            return
        records = validate_pilot_questions(read_jsonl(path))
        report = assess_pilot_questions(records, private_ids=[])
        if args.command == "pilot-manifest" and not report["errors"]:
            manifest = create_pilot_manifest(path, records)
            write_pilot_manifest(path.parent / "pilot_manifest.json", manifest)
            report["manifest"] = manifest
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "pre-dgx-check":
        print(json.dumps(pre_dgx_check(root), ensure_ascii=False, indent=2))
    elif args.command == "validate":
        if not args.file:
            parser.error("validate에는 --file이 필요합니다")
        rows = read_jsonl(Path(args.file))
        validator = validate_questions if "questions" in Path(args.file).parts else validate_responses
        print(f"valid: {len(validator(rows))} records")
    elif args.command == "generate":
        rows = generate_responses(_questions(root))
        write_jsonl(root / "data" / "raw" / "gpt_oss" / "mock_teacher_a.jsonl", [x for x in rows if x["teacher_model"] == "MockTeacherA"])
        write_jsonl(root / "data" / "raw" / "glm" / "mock_teacher_b.jsonl", [x for x in rows if x["teacher_model"] == "MockTeacherB"])
        print(f"generated: {len(rows)} teacher responses")
    elif args.command in {"merge", "conflicts"}:
        rows = merge_responses(_raw(root))
        write_jsonl(root / "data" / "merged" / "mock_merged.jsonl", rows)
        print(json.dumps({"merged": len(rows), "conflicts": {kind: sum(1 for item in rows if item["conflict_type"] == kind) for kind in ["agree", "answer_conflict", "one_missing", "format_conflict", "both_suspect"]}}, ensure_ascii=False))
    elif args.command == "verify":
        rows = verify_responses(_raw(root), _questions(root))
        write_jsonl(root / "data" / "verified" / "mock_verified.jsonl", rows)
        print(f"verified: {sum(1 for row in rows if row['training_approved'])}/{len(rows)} approved")
    elif args.command == "build-conditions":
        verified_path = root / "data" / "verified" / "mock_verified.jsonl"
        conditions = build_conditions(_questions(root), validate_responses(read_jsonl(verified_path)))
        data_conditions = {name: rows for name, rows in conditions.items() if isinstance(rows, list)}
        for name, rows in data_conditions.items():
            write_jsonl(root / "data" / "conditions" / f"{name}.jsonl", rows)
        print(json.dumps({name: len(rows) for name, rows in data_conditions.items()}))
    elif args.command == "mock-e2e":
        print(json.dumps(run_mock_pipeline(root), ensure_ascii=False, indent=2))
    elif args.command == "leakage-check":
        assert_private_final_is_safe(root / "evaluation" / "private_final")
        print("private_final safe: no evaluation items found")
    else:
        print((root / "reports" / "mock_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
