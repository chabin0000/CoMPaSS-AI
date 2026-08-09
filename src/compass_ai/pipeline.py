"""Mock 질문부터 S 조건과 요약까지의 전체 E2E 파이프라인."""

from __future__ import annotations

import json
from pathlib import Path

from .conditions import build_conditions, condition_stats
from .jsonl import read_jsonl, write_jsonl
from .leakage import assert_private_final_is_safe
from .logging_utils import create_run_logger
from .schemas import validate_questions, validate_responses
from .teacher import generate_responses, merge_responses
from .verification import verify_responses


def run_mock_pipeline(root: Path, seed: int = 20260809) -> dict:
    logger = create_run_logger(root / "logs", {"seed": seed, "mode": "mock"}, ["python", "-m", "compass_ai", "mock-e2e"])
    try:
        private = root / "evaluation" / "private_final"
        private.mkdir(parents=True, exist_ok=True)
        assert_private_final_is_safe(private)
        question_path = root / "data" / "questions" / "mock_questions.jsonl"
        # 테스트용 임시 폴더에서도 저장소에 추적된 기본 Mock 질문을 사용한다.
        if not question_path.exists():
            question_path = Path(__file__).resolve().parents[2] / "data" / "questions" / "mock_questions.jsonl"
        questions = validate_questions(read_jsonl(question_path))
        logger.event("questions_loaded", {"count": len(questions)})
        raw = validate_responses(generate_responses(questions))
        write_jsonl(root / "data" / "raw" / "gpt_oss" / "mock_teacher_a.jsonl", [x for x in raw if x["teacher_model"] == "MockTeacherA"])
        write_jsonl(root / "data" / "raw" / "glm" / "mock_teacher_b.jsonl", [x for x in raw if x["teacher_model"] == "MockTeacherB"])
        merged = merge_responses(raw)
        write_jsonl(root / "data" / "merged" / "mock_merged.jsonl", merged)
        verified = validate_responses(verify_responses(raw, questions))
        write_jsonl(root / "data" / "verified" / "mock_verified.jsonl", verified)
        conditions = build_conditions(questions, verified)
        data_conditions = {name: rows for name, rows in conditions.items() if isinstance(rows, list)}
        for name, rows in data_conditions.items():
            write_jsonl(root / "data" / "conditions" / f"{name}.jsonl", rows)
        stats = {name: condition_stats(rows) for name, rows in data_conditions.items()}
        summary = {"research_notice": "Mock 결과는 소프트웨어 파이프라인 구현 검증이며 실제 AI 모델 성능 결과가 아니다.", "question_count": len(questions), "raw_response_count": len(raw), "conflict_counts": {kind: sum(1 for row in merged if row["conflict_type"] == kind) for kind in ["agree", "answer_conflict", "one_missing", "format_conflict", "both_suspect"]}, "condition_counts": {name: len(rows) for name, rows in data_conditions.items()}, "condition_stats": stats, "selection_metadata": conditions["selection_metadata"], "condition_metadata": conditions["condition_metadata"], "run_id": logger.path.name}
        (root / "reports").mkdir(exist_ok=True)
        (root / "reports" / "mock_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.write_metrics(summary)
        logger.event("completed", {"question_count": len(questions)})
        return summary
    except Exception as error:
        logger.event("failed", {"error": repr(error)})
        raise
