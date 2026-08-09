"""실제 파일럿 질문의 작성 전·후 품질 검사와 manifest 생성."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

from .schemas import SchemaError, validate_questions

VALID_VERIFICATION_METHODS = {
    "answer_key", "calculation", "unit_check", "code_execution", "official_source", "human_review",
    "official_source_review", "official_archive_review", "human_rubric", "programmatic_constraints",
    "exact_match", "exact_semantic", "dual_source_review", "rubric_and_constraints",
    "calculation_and_principle", "code_review", "test_rubric", "official_documentation",
    "research_source_review", "source_review",
}
AUTO_VERIFICATION_METHODS = {"answer_key", "calculation", "unit_check", "code_execution", "programmatic_constraints", "exact_match"}
HYBRID_VERIFICATION_METHODS = {"exact_semantic", "rubric_and_constraints", "calculation_and_principle", "test_rubric"}


def _preserve_notes_metadata(item: dict) -> dict:
    """초안의 notes에 있던 연구 provenance를 구조화하되 원문 notes는 남긴다."""
    normalized = dict(item)
    for part in str(item.get("notes", "")).split(";"):
        part = part.strip()
        if "=" not in part:
            if part == "record response_status":
                normalized.setdefault("response_status_required", True)
            continue
        key, value = (piece.strip() for piece in part.split("=", 1))
        if key in {"rubric", "origin", "inspiration"}:
            normalized.setdefault(key, value)
        elif key == "paired_controls":
            normalized.setdefault("paired_control_ids", [identifier for identifier in value.split(",") if identifier])
        elif key == "paired_china":
            normalized.setdefault("paired_china_id", value)
    return normalized


def _grading_mode(item: dict) -> str:
    """검증 방법에 따라 자동 판정과 사람 검수를 구분한다."""
    if item["verification_method"] in AUTO_VERIFICATION_METHODS:
        return "auto"
    if item["verification_method"] in HYBRID_VERIFICATION_METHODS:
        return "hybrid"
    return "human"


def validate_pilot_questions(records: list[dict]) -> list[dict]:
    """Mock과 분리된 실제 파일럿 질문 계약을 검사한다."""
    result = [_preserve_notes_metadata(item) for item in validate_questions(records)]
    for index, item in enumerate(result, start=1):
        if item["split"] != "pilot":
            raise SchemaError(f"pilot {index}: split must be pilot")
        if item["question_id"].lower().startswith("mock-") or "mock_case" in item:
            raise SchemaError(f"pilot {index}: Mock data is not allowed")
        if not str(item["question"]).strip():
            raise SchemaError(f"pilot {index}: question is empty")
        if not str(item["reference_answer"]).strip():
            raise SchemaError(f"pilot {index}: reference_answer is empty")
        if item["verification_method"] not in VALID_VERIFICATION_METHODS:
            raise SchemaError(f"pilot {index}: invalid verification_method")
        if not str(item["evidence_source"]).strip():
            raise SchemaError(f"pilot {index}: evidence_source is required")
        if "paired_control_ids" in item and not all(isinstance(value, str) and value for value in item["paired_control_ids"]):
            raise SchemaError(f"pilot {index}: paired_control_ids must contain question IDs")
        if "paired_china_id" in item and not isinstance(item["paired_china_id"], str):
            raise SchemaError(f"pilot {index}: paired_china_id must be a question ID")
    return result


def _normalize_text(value: str) -> str:
    return "".join(value.lower().split())


def assess_pilot_questions(records: list[dict], private_ids: list[str]) -> dict:
    """오류는 빈 목록일 때만 manifest 생성 단계로 넘어갈 수 있다."""
    errors, warnings = [], []
    private = set(private_ids)
    for item in records:
        if item["question_id"] in private:
            errors.append({"type": "private_final_id_overlap", "question_id": item["question_id"]})
    by_id = {item["question_id"]: item for item in records}
    china = [item for item in records if item["domain"] == "politics_china"]
    controls = [item for item in records if item["domain"] == "politics_control"]
    valid_pairs = 0
    for item in china:
        for control_id in item.get("paired_control_ids", []):
            control = by_id.get(control_id)
            if not control or control.get("paired_china_id") != item["question_id"]:
                errors.append({"type": "invalid_political_pair", "china_question_id": item["question_id"], "control_question_id": control_id})
            else:
                valid_pairs += 1
    for item in controls:
        china_id = item.get("paired_china_id")
        if china_id and (china_id not in by_id or item["question_id"] not in by_id[china_id].get("paired_control_ids", [])):
            errors.append({"type": "invalid_political_pair", "china_question_id": china_id, "control_question_id": item["question_id"]})
    for left_index, left in enumerate(records):
        left_text = _normalize_text(left["question"])
        for right in records[left_index + 1:]:
            right_text = _normalize_text(right["question"])
            ratio = SequenceMatcher(None, left_text, right_text).ratio()
            if ratio >= 0.9:
                warnings.append({"type": "possible_text_duplicate", "question_ids": [left["question_id"], right["question_id"]], "similarity": round(ratio, 3)})
    return {
        "total_questions": len(records), "domains": dict(Counter(row["domain"] for row in records)),
        "task_types": dict(Counter(row["task_type"] for row in records)),
        "difficulty": dict(Counter(row["difficulty"] for row in records)),
        "verification_methods": dict(Counter(row["verification_method"] for row in records)),
        "grading_modes": dict(Counter(_grading_mode(row) for row in records)),
        "requires_current_info": [row["question_id"] for row in records if row["requires_current_info"]],
        "political_pairing": {"china_questions": len(china), "control_questions": len(controls), "valid_pairs": valid_pairs},
        "errors": errors, "warnings": warnings,
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_pilot_manifest(source_file: Path, records: list[dict], dataset_version: str = "pilot-v1") -> dict:
    """동일 질문 집합을 두 교사가 썼는지 확인할 수 있는 manifest를 만든다."""
    ids = sorted(row["question_id"] for row in records)
    return {
        "dataset_name": "compass-ai-pilot", "dataset_version": dataset_version,
        "created_at": datetime.now(UTC).isoformat(), "question_count": len(records),
        "question_id_sha256": _sha256_bytes("\n".join(ids).encode()),
        "source_file_sha256": _sha256_bytes(source_file.read_bytes()),
        "domain_distribution": dict(Counter(row["domain"] for row in records)),
        "task_type_distribution": dict(Counter(row["task_type"] for row in records)),
        "verification_method_distribution": dict(Counter(row["verification_method"] for row in records)),
        "grading_mode_distribution": dict(Counter(_grading_mode(row) for row in records)),
        "requires_current_info_count": sum(bool(row["requires_current_info"]) for row in records),
    }


def write_pilot_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
