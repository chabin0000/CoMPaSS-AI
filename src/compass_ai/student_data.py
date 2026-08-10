"""검증/조건 결과를 ARGOS 학생 학습용 JSONL로 변환한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .jsonl import read_jsonl, write_jsonl
from .leakage import assert_no_evaluation_leakage


CONDITIONS = {
    "S1", "S2", "S3a", "S3b", "S3_matched", "S4",
    "S4_teacher_only", "S4_with_human_correction", "S5",
}
STUDENT_FIELDS = {
    "sample_id", "source_question_id", "prompt", "target", "final_answer",
    "source_teacher", "source_type", "verification_status", "condition", "provenance",
}


class StudentDatasetError(ValueError):
    """학생 학습 데이터 계약 또는 provenance 위반."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_student_dataset(records: list[dict]) -> list[dict]:
    result = list(records)
    sample_ids: set[str] = set()
    question_ids: set[str] = set()
    question_teacher_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(result, 1):
        if not isinstance(row, dict):
            raise StudentDatasetError(f"student sample {index}: object required")
        missing = STUDENT_FIELDS - row.keys()
        if missing:
            raise StudentDatasetError(f"student sample {index}: missing {', '.join(sorted(missing))}")
        for field in ("sample_id", "source_question_id", "prompt", "target", "condition"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise StudentDatasetError(f"student sample {index}: {field} is empty")
        if row["target"] != row["final_answer"]:
            raise StudentDatasetError(f"student sample {index}: target/final_answer mismatch")
        if row["condition"] not in CONDITIONS:
            raise StudentDatasetError(f"student sample {index}: unknown condition")
        if row["sample_id"] in sample_ids:
            raise StudentDatasetError(f"duplicate sample_id: {row['sample_id']}")
        sample_ids.add(row["sample_id"])
        pair = (row["source_question_id"], str(row["source_teacher"]))
        if pair in question_teacher_pairs:
            raise StudentDatasetError(f"duplicate question/teacher sample: {pair[0]}")
        question_teacher_pairs.add(pair)
        if row["condition"] != "S3a" and row["source_question_id"] in question_ids:
            raise StudentDatasetError(f"duplicate source_question_id: {row['source_question_id']}")
        question_ids.add(row["source_question_id"])
        if not isinstance(row["provenance"], dict):
            raise StudentDatasetError(f"student sample {index}: provenance must be an object")
    return result


def _assert_source_allowed(row: dict, index: int, condition: str) -> None:
    required = {"question_id", "question", "final_answer"}
    missing = required - row.keys()
    if missing:
        raise StudentDatasetError(f"source sample {index}: missing {', '.join(sorted(missing))}")
    if row.get("dataset_version") == "pilot-v1":
        raise StudentDatasetError("Pilot-v1 is not student training data")
    question_id = str(row.get("question_id", ""))
    if row.get("split") in {"validation", "private_eval"} or question_id.startswith(("CORE-VA-", "CORE-PE-", "CORE2-VA-", "CORE2-PE-")):
        raise StudentDatasetError(f"{row.get('split') or question_id} is not student training data")
    if "reference_answer" in row or "rubric" in row:
        raise StudentDatasetError("verification reference/rubric material must not enter student data")
    if row.get("tag") == "excluded" or row.get("verification_outcome") == "excluded":
        raise StudentDatasetError(f"source sample {index}: rejected sample")
    if row.get("error_type"):
        raise StudentDatasetError(f"source sample {index}: error sample")
    independently_resolved = (
        row.get("verification_result") == "verified"
        and row.get("verification_status") == "passed"
        and row.get("training_approved") is True
    )
    if row.get("conflict_type") not in {None, "agree"} and not independently_resolved:
        raise StudentDatasetError(f"source sample {index}: conflict sample")
    if row.get("response_status") in {"refusal", "no_answer"}:
        raise StudentDatasetError(f"source sample {index}: response_status is not trainable")
    if not str(row["question"]).strip() or not str(row["final_answer"]).strip():
        raise StudentDatasetError(f"source sample {index}: prompt or final answer is empty")
    if condition.startswith("S4") and (
        row.get("verification_status") != "passed" or not row.get("training_approved")
    ):
        raise StudentDatasetError(f"source sample {index}: S4 requires passed and approved verification")


def build_student_dataset(
    source_path: Path,
    output_path: Path,
    condition: str,
    evaluation_path: Path | None = None,
) -> dict:
    """원문 reasoning을 버리고 final answer와 최소 provenance만 분리 저장한다."""
    if condition not in CONDITIONS:
        raise StudentDatasetError(f"unsupported condition: {condition}")
    if "pilot" in {part.lower() for part in source_path.parts}:
        raise StudentDatasetError("Pilot-v1 path is forbidden for student training")
    lowered_parts = {part.lower() for part in source_path.parts}
    if "evaluation" in lowered_parts or "private_final" in lowered_parts:
        raise StudentDatasetError("evaluation/private-final data is forbidden for student training")
    if output_path.exists():
        raise StudentDatasetError(f"output already exists: {output_path}")
    source_checksum = _file_sha256(source_path)
    samples: list[dict] = []
    for index, row in enumerate(read_jsonl(source_path), 1):
        _assert_source_allowed(row, index, condition)
        teacher = row.get("teacher_model") or row.get("source_teacher")
        source_type = row.get("source_type") or ("teacher" if teacher else "korean_source")
        source_name = str(teacher or source_type)
        digest = hashlib.sha256(
            f"{condition}\n{row['question_id']}\n{source_name}".encode("utf-8")
        ).hexdigest()[:16]
        provenance_keys = (
            "checkpoint", "revision", "quantization", "generation_config", "evidence",
            "verification_outcome", "tag", "reviewer", "timestamp", "condition_selection",
            "domain", "task_type", "dataset_checksum", "run_id", "verification_method",
            "verification_result", "conflict_type", "teacher_a_raw_response_reference",
            "teacher_b_raw_response_reference", "rejection_reason", "verifier_metadata",
            "source_run_ids",
        )
        provenance = {key: row[key] for key in provenance_keys if key in row}
        provenance.update({"source_file": str(source_path), "source_file_checksum": source_checksum})
        samples.append({
            "sample_id": f"{condition}-{digest}",
            "source_question_id": row["question_id"],
            "prompt": row["question"],
            "target": row["final_answer"],
            "final_answer": row["final_answer"],
            "source_teacher": teacher,
            "source_type": source_type,
            "verification_status": row.get("verification_status", "pending"),
            "condition": condition,
            "provenance": provenance,
        })
    validate_student_dataset(samples)
    if not samples:
        raise StudentDatasetError(f"{condition} has no trainable samples")
    if evaluation_path is not None:
        evaluation_ids = [row["evaluation_id"] for row in read_jsonl(evaluation_path)]
        assert_no_evaluation_leakage([row["source_question_id"] for row in samples], evaluation_ids)
    write_jsonl(output_path, samples)
    dataset_checksum = _file_sha256(output_path)
    metadata = {
        "condition": condition,
        "sample_count": len(samples),
        "source_checksum": source_checksum,
        "dataset_checksum": dataset_checksum,
        "question_ids": sorted({row["source_question_id"] for row in samples}),
    }
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {**metadata, "output_path": str(output_path)}
