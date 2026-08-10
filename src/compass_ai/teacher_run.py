"""실제 교사 모델 실행 전 계약과 결과 비교 준비 검사."""

from __future__ import annotations

from pathlib import Path

from .jsonl import read_jsonl
from .schemas import SchemaError, VALID_RESPONSE_STATUSES


CONFIG_FIELDS = {"experiment_id", "dataset_name", "dataset_version", "dataset_checksum", "teacher_model", "checkpoint", "revision", "quantization", "inference_engine", "system_prompt", "temperature", "top_p", "max_new_tokens", "seed", "num_repeats", "timestamp", "environment_log_reference", "status"}
RAW_FIELDS = {"question_id", "teacher_model", "dataset_version", "dataset_checksum", "run_id", "raw_response", "final_answer", "generation_config_reference", "response_started_at", "response_finished_at", "latency_seconds", "ttft_seconds", "input_tokens", "output_tokens", "response_status", "error", "metadata"}


def validate_teacher_config(config: dict) -> dict:
    missing = CONFIG_FIELDS - config.keys()
    if missing:
        raise SchemaError(f"teacher config: 필수 필드 누락: {', '.join(sorted(missing))}")
    if config["dataset_name"] != "compass-ai-pilot" or config["dataset_version"] != "pilot-v1":
        raise SchemaError("teacher config: Pilot-v1 dataset 정보가 일치하지 않습니다")
    if len(config["dataset_checksum"]) != 64 or config["status"] not in {"proposed", "final"}:
        raise SchemaError("teacher config: checksum 또는 status가 올바르지 않습니다")
    return config


def validate_teacher_raw_responses(records: list[dict]) -> list[dict]:
    result = list(records)
    for index, row in enumerate(result, 1):
        missing = RAW_FIELDS - row.keys()
        if missing:
            raise SchemaError(f"teacher raw {index}: 필수 필드 누락: {', '.join(sorted(missing))}")
        if row["teacher_model"].lower().startswith("mock"):
            raise SchemaError(f"teacher raw {index}: Mock 결과는 실제 교사 결과가 아닙니다")
        if row["response_status"] not in VALID_RESPONSE_STATUSES:
            raise SchemaError(f"teacher raw {index}: 알 수 없는 response_status입니다")
    return result


def resume_question_ids(records: list[dict]) -> set[str]:
    """오류 없이 저장된 응답만 재개 시 건너뛴다."""
    return {row["question_id"] for row in records if row["error"] is None and row["response_status"] != "no_answer"}


def comparison_readiness(gpt_path: Path, glm_path: Path, expected_dataset_checksum: str) -> dict:
    """실제 두 교사 raw JSONL이 같은 실험 조건인지 검사한다."""
    if not gpt_path.exists() or not glm_path.exists():
        return {"status": "not_ready", "errors": [], "reason": "teacher results missing"}
    try:
        gpt, glm = validate_teacher_raw_responses(read_jsonl(gpt_path)), validate_teacher_raw_responses(read_jsonl(glm_path))
    except (SchemaError, ValueError) as error:
        return {"status": "error", "errors": [str(error)]}
    errors = []
    if {row["dataset_checksum"] for row in [*gpt, *glm]} != {expected_dataset_checksum}:
        errors.append("dataset_checksum_mismatch")
    if {row["question_id"] for row in gpt} != {row["question_id"] for row in glm}:
        errors.append("question_id_mismatch")
    if {row["generation_config_reference"] for row in gpt} != {row["generation_config_reference"] for row in glm}:
        errors.append("generation_config_mismatch")
    if len(gpt) != len(glm):
        errors.append("question_count_mismatch")
    metadata_recorded = all(
        isinstance(row.get("metadata"), dict)
        and bool(row["metadata"].get("full_config_checksum"))
        and isinstance(row["metadata"].get("model_specific"), dict)
        for row in [*gpt, *glm]
    )
    if not metadata_recorded:
        errors.append("model_specific_metadata_missing")
    return {
        "status": "error" if errors else "ready",
        "errors": errors,
        "gpt_count": len(gpt),
        "glm_count": len(glm),
        "model_specific_settings_recorded": metadata_recorded,
    }
