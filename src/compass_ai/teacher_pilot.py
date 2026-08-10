"""동결 Pilot-v1을 실제 OpenAI-compatible 교사 API로 순차 실행한다."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .jsonl import read_jsonl
from .pilot import create_pilot_manifest, validate_pilot_questions
from .teacher_api import OpenAICompatibleTeacher, TeacherAPIError, load_teacher_api_config
from .teacher_run import resume_question_ids, validate_teacher_raw_responses


class DatasetChecksumError(ValueError):
    """동결 dataset 또는 manifest checksum 불일치."""


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_frozen_pilot(root: Path, expected_checksum: str) -> tuple[list[dict], dict]:
    pilot_dir = root / "data" / "questions" / "pilot"
    questions_path = pilot_dir / "pilot_questions.jsonl"
    manifest_path = pilot_dir / "pilot_manifest.json"
    if not questions_path.exists() or not manifest_path.exists():
        raise DatasetChecksumError("frozen Pilot-v1 dataset or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_checksum = hashlib.sha256(questions_path.read_bytes()).hexdigest()
    if actual_checksum != manifest.get("source_file_sha256") or actual_checksum != expected_checksum:
        raise DatasetChecksumError(
            f"Pilot-v1 checksum mismatch: config={expected_checksum}, manifest={manifest.get('source_file_sha256')}, actual={actual_checksum}"
        )
    questions = validate_pilot_questions(read_jsonl(questions_path))
    generated = create_pilot_manifest(questions_path, questions)
    stable_fields = set(generated) - {"created_at"}
    mismatched = [field for field in sorted(stable_fields) if manifest.get(field) != generated.get(field)]
    if mismatched:
        raise DatasetChecksumError(f"Pilot-v1 manifest mismatch: {', '.join(mismatched)}")
    return questions, manifest


def _append_jsonl(path: Path, row: dict) -> None:
    """각 문항 직후 디스크에 append하고 flush하여 중단 시 결과를 보존한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run_teacher_pilot(root: Path, teacher_name: str, config_path: Path, output_root: Path | None = None) -> dict:
    """동결 질문을 순서대로 실행하고 성공 ID만 자동 resume에서 건너뛴다."""
    config = load_teacher_api_config(config_path)
    if teacher_name != config["teacher_name"]:
        raise ValueError(f"teacher mismatch: CLI={teacher_name}, config={config['teacher_name']}")
    questions, manifest = load_frozen_pilot(root, config["dataset_checksum"])
    full_config_checksum = _sha256_json(config)
    common_generation = {"system_prompt": config["system_prompt"], "generation": config["generation"]}
    generation_checksum = _sha256_json(common_generation)
    base = output_root or root / "data" / "raw" / "teacher_runs"
    run_dir = base / teacher_name / config["experiment_id"]
    responses_path = run_dir / "responses.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metadata_path = run_dir / "run.json"
    now = datetime.now(UTC).isoformat()
    if metadata_path.exists():
        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if run_metadata["full_config_checksum"] != full_config_checksum:
            raise ValueError("resume config checksum mismatch; use a new experiment_id")
        run_metadata["resume_attempts"] = int(run_metadata.get("resume_attempts", 0)) + 1
    else:
        run_metadata = {
            "run_id": f"{teacher_name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}",
            "teacher_name": teacher_name,
            "teacher_model": config["model"],
            "model_metadata": config["model_metadata"],
            "dataset_name": manifest["dataset_name"],
            "dataset_version": manifest["dataset_version"],
            "dataset_checksum": manifest["source_file_sha256"],
            "generation_config_checksum": generation_checksum,
            "full_config_checksum": full_config_checksum,
            "started_at": now,
            "resume_attempts": 0,
        }
    _write_metadata(metadata_path, run_metadata)
    existing = validate_teacher_raw_responses(read_jsonl(responses_path)) if responses_path.exists() else []
    completed_ids = resume_question_ids(existing)
    client = OpenAICompatibleTeacher(config)
    for question in questions:
        if question["question_id"] in completed_ids:
            continue
        started_at = datetime.now(UTC).isoformat()
        try:
            answer = client.generate(question)
            finished_at = datetime.now(UTC).isoformat()
            row = {
                "question_id": question["question_id"],
                "teacher_model": config["model"],
                "dataset_version": manifest["dataset_version"],
                "dataset_checksum": manifest["source_file_sha256"],
                "run_id": run_metadata["run_id"],
                "raw_response": answer.raw_response,
                "final_answer": answer.final_answer,
                "generation_config_reference": generation_checksum,
                "response_started_at": started_at,
                "response_finished_at": finished_at,
                "latency_seconds": answer.latency_seconds,
                "ttft_seconds": None,
                "tokens_per_second": None,
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
                "response_status": "answered",
                "error": None,
                "metadata": {
                    "teacher_name": teacher_name,
                    "model": config["model"],
                    "full_config_checksum": full_config_checksum,
                    "model_specific": {**config["model_metadata"], "extra_body": config["extra_body"]},
                },
            }
            _append_jsonl(responses_path, row)
            completed_ids.add(question["question_id"])
        except TeacherAPIError as error:
            _append_jsonl(errors_path, {
                "run_id": run_metadata["run_id"],
                "question_id": question["question_id"],
                "timestamp": datetime.now(UTC).isoformat(),
                "error_type": error.error_type,
                "error": str(error),
            })
    completed_count = len(completed_ids)
    status = "complete" if completed_count == len(questions) else "incomplete"
    run_metadata.update({
        "status": status,
        "completed_count": completed_count,
        "expected_count": len(questions),
        "last_attempt_finished_at": datetime.now(UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat() if status == "complete" else None,
    })
    _write_metadata(metadata_path, run_metadata)
    return {
        **run_metadata,
        "responses_path": str(responses_path),
        "errors_path": str(errors_path),
        "run_metadata_path": str(metadata_path),
    }
