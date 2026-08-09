"""연구 데이터 계약의 최소 검증 도구."""

from __future__ import annotations

from typing import Iterable


class SchemaError(ValueError):
    """JSONL 레코드가 연구 데이터 계약을 지키지 않을 때 발생한다."""


QUESTION_FIELDS = {
    "question_id", "split", "domain", "task_type", "question", "reference_answer",
    "verification_method", "evidence_source", "difficulty", "requires_current_info", "notes",
}
RESPONSE_FIELDS = {
    "question_id", "domain", "task_type", "question", "teacher_model", "checkpoint",
    "quantization", "generation_config", "raw_response", "final_answer",
    "verification_status", "verification_method", "evidence", "error_type", "tag",
    "reviewer", "verification_outcome", "timestamp", "training_approved",
}
VALID_TAGS = {"teacher_raw", "teacher_verified", "human_corrected", "korean_source", "excluded"}
VALID_RESPONSE_STATUSES = {"answered", "partial", "evasive", "refusal", "no_answer"}


def _validate(records: Iterable[dict], required: set[str], label: str) -> list[dict]:
    result = list(records)
    ids: set[str] = set()
    for index, record in enumerate(result, start=1):
        if not isinstance(record, dict):
            raise SchemaError(f"{label} {index}: object여야 합니다")
        missing = required - record.keys()
        if missing:
            raise SchemaError(f"{label} {index}: 필수 필드 누락: {', '.join(sorted(missing))}")
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise SchemaError(f"{label} {index}: question_id가 필요합니다")
        # 교사 응답은 동일 질문에 교사별로 여러 개가 가능하다.
        key = question_id if label == "question" else f"{question_id}:{record.get('teacher_model')}"
        if key in ids:
            raise SchemaError(f"{label} {index}: duplicate question_id 또는 교사 응답입니다: {question_id}")
        ids.add(key)
        if label == "response" and record["tag"] not in VALID_TAGS:
            raise SchemaError(f"response {index}: 알 수 없는 tag입니다")
        if label == "response" and "response_status" in record and record["response_status"] not in VALID_RESPONSE_STATUSES:
            raise SchemaError(f"response {index}: 알 수 없는 response_status입니다")
    return result


def validate_questions(records: Iterable[dict]) -> list[dict]:
    return _validate(records, QUESTION_FIELDS, "question")


def validate_responses(records: Iterable[dict]) -> list[dict]:
    return _validate(records, RESPONSE_FIELDS, "response")
