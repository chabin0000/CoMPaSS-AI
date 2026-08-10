"""Pilot과 분리된 Core-v2 공개 train 문항을 같은 teacher API 계약으로 실행한다."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .jsonl import read_jsonl
from .teacher_api import OpenAICompatibleTeacher, TeacherAPIError, load_teacher_api_config
from .teacher_pilot import _append_jsonl, _sha256_json, _write_metadata
from .teacher_run import resume_question_ids, validate_teacher_raw_responses


def run_teacher_dataset(root: Path, teacher_name: str, config_path: Path, questions_path: Path, output_root: Path | None = None) -> dict:
    config = load_teacher_api_config(config_path)
    if config["dataset_version"] != "core-v2" or config["dataset_name"] != "compass-ai-core":
        raise ValueError("run-teacher-dataset requires a Core-v2 teacher config")
    if teacher_name != config["teacher_name"]:
        raise ValueError(f"teacher mismatch: CLI={teacher_name}, config={config['teacher_name']}")
    actual_checksum = hashlib.sha256(questions_path.read_bytes()).hexdigest()
    if actual_checksum != config["dataset_checksum"]:
        raise ValueError("Core-v2 teacher source checksum mismatch")
    questions = read_jsonl(questions_path)
    if len(questions) != 120 or any(row.get("dataset_version") != "core-v2" or row.get("split") != "train" for row in questions):
        raise ValueError("Core-v2 teacher source must be the frozen 120-item train split")

    full_config_checksum = _sha256_json(config)
    generation_checksum = _sha256_json({"system_prompt": config["system_prompt"], "generation": config["generation"]})
    base = output_root or root / "data" / "raw" / "teacher_runs"
    run_dir = base / teacher_name / config["experiment_id"]
    responses_path, errors_path, metadata_path = run_dir / "responses.jsonl", run_dir / "errors.jsonl", run_dir / "run.json"
    now = datetime.now(UTC).isoformat()
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["full_config_checksum"] != full_config_checksum:
            raise ValueError("resume config checksum mismatch; use a new experiment_id")
        metadata["resume_attempts"] = int(metadata.get("resume_attempts", 0)) + 1
    else:
        metadata = {
            "run_id": f"{teacher_name}-core-v2-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}",
            "teacher_name": teacher_name, "teacher_model": config["model"], "model_metadata": config["model_metadata"],
            "dataset_name": config["dataset_name"], "dataset_version": "core-v2",
            "dataset_checksum": actual_checksum, "generation_config_checksum": generation_checksum,
            "full_config_checksum": full_config_checksum, "started_at": now, "resume_attempts": 0,
        }
    _write_metadata(metadata_path, metadata)
    existing = validate_teacher_raw_responses(read_jsonl(responses_path)) if responses_path.exists() else []
    completed = resume_question_ids(existing)
    client = OpenAICompatibleTeacher(config)
    for question in questions:
        if question["question_id"] in completed:
            continue
        started_at = datetime.now(UTC).isoformat()
        try:
            answer = client.generate(question)
            row = {
                "question_id": question["question_id"], "teacher_model": config["model"],
                "dataset_version": "core-v2", "dataset_checksum": actual_checksum, "run_id": metadata["run_id"],
                "raw_response": answer.raw_response, "final_answer": answer.final_answer,
                "generation_config_reference": generation_checksum, "response_started_at": started_at,
                "response_finished_at": datetime.now(UTC).isoformat(), "latency_seconds": answer.latency_seconds,
                "ttft_seconds": None, "tokens_per_second": None, "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens, "response_status": "answered", "error": None,
                "metadata": {"teacher_name": teacher_name, "model": config["model"], "full_config_checksum": full_config_checksum,
                             "model_specific": {**config["model_metadata"], "extra_body": config["extra_body"]}},
            }
            _append_jsonl(responses_path, row); completed.add(question["question_id"])
        except TeacherAPIError as error:
            _append_jsonl(errors_path, {"run_id": metadata["run_id"], "question_id": question["question_id"],
                         "timestamp": datetime.now(UTC).isoformat(), "error_type": error.error_type, "error": str(error)})
    status = "complete" if len(completed) == len(questions) else "incomplete"
    metadata.update({"status": status, "completed_count": len(completed), "expected_count": len(questions),
                     "last_attempt_finished_at": datetime.now(UTC).isoformat(),
                     "completed_at": datetime.now(UTC).isoformat() if status == "complete" else None})
    _write_metadata(metadata_path, metadata)
    return {**metadata, "responses_path": str(responses_path), "errors_path": str(errors_path), "run_metadata_path": str(metadata_path)}
