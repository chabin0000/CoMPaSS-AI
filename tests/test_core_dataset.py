import json
from pathlib import Path

import pytest

from compass_ai.core_dataset import CoreDatasetError, check_core_dataset
from compass_ai.jsonl import read_jsonl
from compass_ai.student_data import StudentDatasetError, build_student_dataset


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "data" / "questions" / "core_v1"


def test_core_v1_counts_balance_disjointness_and_manifest():
    report = check_core_dataset(ROOT)
    assert report["status"] == "ready"
    assert report["split_counts"] == {"train": 120, "validation": 24, "private_eval": 48}
    assert report["question_count"] == 192
    assert max(report["domain_counts"].values()) - min(report["domain_counts"].values()) <= 2
    assert report["auto_verifiable_count"] / report["question_count"] >= 0.8
    assert len(report["dataset_checksum"]) == 64


def test_core_v1_private_prompts_do_not_expose_answers_or_rubric_evidence():
    rows = read_jsonl(CORE / "private_eval_questions.jsonl")
    assert len(rows) == 48
    for row in rows:
        assert row["split"] == "private_eval"
        assert row["reference"]["visibility"] == "private"
        assert set(row["reference"]) == {"visibility", "key_id", "answer_sha256"}
        assert row["rubric"]["visibility"] == "private"
        serialized = json.dumps(row, ensure_ascii=False).lower()
        assert "reference_answer" not in serialized
        assert '"answer"' not in serialized
        assert '"criteria"' not in serialized
        assert '"evidence"' not in serialized


def test_core_v1_ids_and_questions_are_disjoint_from_pilot():
    core_rows = [
        row
        for name in ("train_questions.jsonl", "validation_questions.jsonl", "private_eval_questions.jsonl")
        for row in read_jsonl(CORE / name)
    ]
    pilot_rows = read_jsonl(ROOT / "data" / "questions" / "pilot" / "pilot_questions.jsonl")
    assert not ({row["question_id"] for row in core_rows} & {row["question_id"] for row in pilot_rows})
    normalized = [" ".join(row["question"].casefold().split()) for row in core_rows]
    assert len(normalized) == len(set(normalized))


def test_core_v1_detects_checksum_tampering(tmp_path):
    target = tmp_path / "repo"
    core = target / "data" / "questions" / "core_v1"
    core.mkdir(parents=True)
    for source in CORE.iterdir():
        if source.is_file():
            (core / source.name).write_bytes(source.read_bytes())
    with (core / "train_questions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(CoreDatasetError, match="checksum"):
        check_core_dataset(target)


def test_student_builder_rejects_private_eval_even_outside_evaluation_path(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({
        "question_id": "CORE-PE-MATH-001", "dataset_version": "core-v1",
        "split": "private_eval", "question": "비공개 평가 질문",
        "final_answer": "leaked", "verification_status": "passed",
        "training_approved": True,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(StudentDatasetError, match="private_eval"):
        build_student_dataset(source, tmp_path / "student.jsonl", "S1")


def test_student_builder_rejects_core_validation_by_id_when_split_metadata_is_missing(tmp_path):
    source = tmp_path / "verified.jsonl"
    source.write_text(json.dumps({
        "question_id": "CORE-VA-MATH-001", "dataset_version": "core-v1",
        "question": "검증 질문", "final_answer": "4",
        "verification_status": "passed", "training_approved": True,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(StudentDatasetError, match="CORE-VA"):
        build_student_dataset(source, tmp_path / "student.jsonl", "S1")


def test_condition_area_contains_schema_placeholder_only():
    condition_dir = ROOT / "data" / "student" / "core_v1" / "conditions"
    assert (condition_dir / "condition_dataset.schema.json").exists()
    assert (condition_dir / "condition_manifest.template.json").exists()
    assert not list(condition_dir.glob("*.jsonl"))
