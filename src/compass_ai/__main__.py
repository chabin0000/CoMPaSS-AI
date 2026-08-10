"""`python -m compass_ai` 명령행 인터페이스."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

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
from .teacher_api import teacher_check
from .teacher_pilot import run_teacher_pilot
from .teacher_run import comparison_readiness
from .student_data import build_student_dataset
from .student_training import evaluate_saved_student, run_student
from .student_tiny import run_tiny_student_dry_run
from .student_preflight import student_preflight
from .dgx import build_dgx_plan, collect_dgx_system_report, format_dgx_plan, run_dgx_launcher
from .teacher_verification import apply_verification_review, verify_teacher_runs, verification_readiness
from .rehearsal import run_full_rehearsal
from .core_dataset import check_core_dataset
from .core_v2_dataset import check_core_v2_dataset
from .core_v2_rehearsal import run_core_v2_synthetic_rehearsal
from .pre_dgx_final import pre_dgx_final
from .result_analysis import analyze_results
from .teacher_dataset import run_teacher_dataset
from .core_conditions import build_core_condition_datasets
from .core_evaluation import collect_core_scored_results, prepare_core_private_evaluation


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
    parser.add_argument("command", choices=["system-check", "dgx-readiness", "pilot-check", "pilot-manifest", "dataset-check", "pre-dgx-check", "pre-dgx-final", "student-preflight", "dgx-system-report", "dgx-plan", "dgx-launch", "full-rehearsal", "core-v2-rehearsal", "prepare-core-evaluation", "collect-core-results", "analyze-results", "teacher-check", "run-pilot", "run-teacher-dataset", "comparison-readiness", "verification-readiness", "apply-verification-review", "build-student-data", "build-core-conditions", "train-student", "evaluate-student", "student-dry-run", "validate", "generate", "merge", "conflicts", "verify", "build-conditions", "mock-e2e", "leakage-check", "summary"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--file")
    parser.add_argument("--teacher")
    parser.add_argument("--config")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--dataset-checksum")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--condition")
    parser.add_argument("--evaluation-file")
    parser.add_argument("--runtime-model-check", action="store_true")
    parser.add_argument("--report-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--state-root")
    parser.add_argument("--questions")
    parser.add_argument("--verification-file")
    parser.add_argument("--eligible-file")
    parser.add_argument("--review-file")
    parser.add_argument("--dataset-version", default="core-v1")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--status-report")
    parser.add_argument("--teacher-eligible")
    parser.add_argument("--human-eligible")
    parser.add_argument("--korean-source")
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
    elif args.command == "dataset-check":
        checker = check_core_v2_dataset if args.dataset_version == "core-v2" else check_core_dataset
        print(json.dumps(checker(root), ensure_ascii=False, indent=2))
    elif args.command == "pre-dgx-check":
        print(json.dumps(pre_dgx_check(root, run_tests=not args.skip_tests), ensure_ascii=False, indent=2))
    elif args.command == "pre-dgx-final":
        print(json.dumps(pre_dgx_final(root), ensure_ascii=False, indent=2))
    elif args.command == "student-preflight":
        if not args.config:
            parser.error("student-preflight requires --config")
        print(json.dumps(student_preflight(root, Path(args.config), args.runtime_model_check), ensure_ascii=False, indent=2))
    elif args.command == "dgx-system-report":
        report_dir = Path(args.report_dir) if args.report_dir else root / "reports" / "dgx" / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        print(json.dumps(collect_dgx_system_report(root, report_dir), ensure_ascii=False, indent=2))
    elif args.command == "dgx-plan":
        config_path = Path(args.config or "configs/dgx/argos_v1.yaml")
        print(format_dgx_plan(build_dgx_plan(root, config_path)))
    elif args.command == "dgx-launch":
        config_path = Path(args.config or "configs/dgx/argos_v1.yaml")
        run_id = args.run_id or datetime.now(UTC).strftime("argos-%Y%m%dT%H%M%SZ")
        state_root = Path(args.state_root) if args.state_root else None
        print(json.dumps(run_dgx_launcher(root, config_path, run_id, state_root), ensure_ascii=False, indent=2))
    elif args.command == "full-rehearsal":
        output_root = Path(args.output_root) if args.output_root else root / "artifacts" / "full_rehearsal" / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        print(json.dumps(run_full_rehearsal(root, output_root), ensure_ascii=False, indent=2))
    elif args.command == "core-v2-rehearsal":
        output_root = Path(args.output_root) if args.output_root else root / "artifacts" / "dgx_runs" / f"synthetic-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        status_report = Path(args.status_report) if args.status_report else root / "reports" / "core_v2_synthetic_rehearsal_status.json"
        print(json.dumps(run_core_v2_synthetic_rehearsal(root, output_root, status_report), ensure_ascii=False, indent=2))
    elif args.command == "analyze-results":
        if not args.input or not args.output:
            parser.error("analyze-results requires --input and --output")
        print(json.dumps(analyze_results(Path(args.input), Path(args.output), args.allow_synthetic), ensure_ascii=False, indent=2))
    elif args.command == "prepare-core-evaluation":
        if not args.config or not args.output_root:
            parser.error("prepare-core-evaluation requires --config (snapshot directory) and --output-root")
        print(json.dumps(prepare_core_private_evaluation(root, Path(args.config), Path(args.output_root)), ensure_ascii=False, indent=2))
    elif args.command == "collect-core-results":
        if not args.config or not args.evaluation_file or not args.output_root or not args.run_id:
            parser.error("collect-core-results requires --config, --evaluation-file, --output-root, and --run-id")
        review = Path(args.review_file) if args.review_file else None
        print(json.dumps(collect_core_scored_results(Path(args.config), Path(args.evaluation_file), Path(args.output_root), args.run_id, review), ensure_ascii=False, indent=2))
    elif args.command == "teacher-check":
        if not args.config:
            parser.error("teacher-check requires --config")
        output_root = Path(args.output_root) if args.output_root else None
        questions_path = Path(args.questions) if args.questions else None
        print(json.dumps(teacher_check(root, Path(args.config), smoke=args.smoke, output_root=output_root, questions_path=questions_path), ensure_ascii=False, indent=2))
    elif args.command == "run-pilot":
        if not args.teacher or not args.config:
            parser.error("run-pilot requires --teacher and --config")
        output_root = Path(args.output_root) if args.output_root else None
        print(json.dumps(run_teacher_pilot(root, args.teacher, Path(args.config), output_root=output_root), ensure_ascii=False, indent=2))
    elif args.command == "run-teacher-dataset":
        if not args.teacher or not args.config or not args.questions:
            parser.error("run-teacher-dataset requires --teacher, --config, and --questions")
        output_root = Path(args.output_root) if args.output_root else None
        print(json.dumps(run_teacher_dataset(root, args.teacher, Path(args.config), Path(args.questions), output_root), ensure_ascii=False, indent=2))
    elif args.command == "comparison-readiness":
        if not args.left or not args.right or not args.dataset_checksum:
            parser.error("comparison-readiness requires --left, --right, and --dataset-checksum")
        print(json.dumps(comparison_readiness(Path(args.left), Path(args.right), args.dataset_checksum), ensure_ascii=False, indent=2))
    elif args.command == "verification-readiness":
        if args.verification_file or args.eligible_file:
            if not args.verification_file or not args.eligible_file:
                parser.error("existing verification check requires --verification-file and --eligible-file")
            report = verification_readiness(Path(args.verification_file), Path(args.eligible_file))
        else:
            if not args.left or not args.right or not args.output_root or not args.dataset_checksum:
                parser.error("verification-readiness requires --left, --right, --output-root, and --dataset-checksum")
            questions_path = Path(args.questions) if args.questions else _pilot_file(root)
            report = verify_teacher_runs(
                questions_path, Path(args.left), Path(args.right), Path(args.output_root), args.dataset_checksum
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "apply-verification-review":
        if not args.questions or not args.verification_file or not args.eligible_file or not args.review_file or not args.output_root:
            parser.error("apply-verification-review requires --questions, --verification-file, --eligible-file, --review-file, and --output-root")
        print(json.dumps(apply_verification_review(
            Path(args.questions), Path(args.verification_file), Path(args.eligible_file),
            Path(args.review_file), Path(args.output_root),
        ), ensure_ascii=False, indent=2))
    elif args.command == "build-student-data":
        if not args.input or not args.output or not args.condition:
            parser.error("build-student-data requires --input, --output, and --condition")
        evaluation_path = Path(args.evaluation_file) if args.evaluation_file else None
        print(json.dumps(build_student_dataset(Path(args.input), Path(args.output), args.condition, evaluation_path), ensure_ascii=False, indent=2))
    elif args.command == "build-core-conditions":
        required = [args.left, args.right, args.teacher_eligible, args.human_eligible, args.output_root]
        if not all(required):
            parser.error("build-core-conditions requires --left, --right, --teacher-eligible, --human-eligible, and --output-root")
        korean_source = Path(args.korean_source) if args.korean_source else None
        print(json.dumps(build_core_condition_datasets(
            root, Path(args.left), Path(args.right), Path(args.teacher_eligible),
            Path(args.human_eligible), korean_source, Path(args.output_root),
        ), ensure_ascii=False, indent=2))
    elif args.command == "train-student":
        if not args.config:
            parser.error("train-student requires --config")
        print(json.dumps(run_student(Path(args.config), root), ensure_ascii=False, indent=2))
    elif args.command == "evaluate-student":
        if not args.config:
            parser.error("evaluate-student requires --config")
        print(json.dumps(evaluate_saved_student(Path(args.config), root), ensure_ascii=False, indent=2))
    elif args.command == "student-dry-run":
        if not args.config:
            parser.error("student-dry-run requires --config")
        print(json.dumps(run_tiny_student_dry_run(Path(args.config), root), ensure_ascii=False, indent=2))
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
