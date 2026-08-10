import hashlib
import json
from pathlib import Path

from compass_ai.core_v2_dataset import check_core_v2_dataset
from compass_ai.jsonl import read_jsonl
from compass_ai.student_data import StudentDatasetError, build_student_dataset
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_core_v2_frozen_dataset_and_protocol_are_ready():
    report = check_core_v2_dataset(ROOT)
    assert report["status"] == "ready"
    assert report["split_counts"] == {"train": 120, "validation": 24, "private_eval": 48}
    assert report["quality_audit_status_counts"] == {"FAIL": 0, "PASS": 190, "REVIEW": 2}
    assert report["review_acceptance"] == {
        "decision": "accepted_without_dataset_change",
        "question_ids": ["CORE2-PE-LOG-006", "CORE2-TR-LOG-010"],
    }
    assert report["near_duplicate_template_groups"] == 0
    assert report["cross_split_template_leakage_pairs"] == 0
    assert report["auto_verifiable_ratio"] >= 0.85
    assert len(report["evaluation_protocol_checksum"]) == 64


def test_core_v2_private_eval_exposes_no_answer_or_rubric_criteria():
    rows = read_jsonl(ROOT / "data" / "questions" / "core_v2" / "private_eval_questions.jsonl")
    assert len(rows) == 48
    for row in rows:
        assert set(row["reference"]) == {"visibility", "key_id", "answer_sha256"}
        assert row["reference"]["visibility"] == "private"
        assert "criteria" not in row["rubric"]
        serialized = json.dumps(row, ensure_ascii=False).lower()
        assert '"answer"' not in serialized
        assert "reference_answer" not in serialized


def test_core_v1_files_remain_at_audited_draft_hashes():
    core = ROOT / "data" / "questions" / "core_v1"
    expected = {
        "train_questions.jsonl": "41b37db3b66dfd27c21d6b016a1539984169c56d8f2fdac9e4477c3280d2532f",
        "validation_questions.jsonl": "3e0646227da6469213678ca4889f3b296c41c86a9835d278720080562a66bb04",
        "private_eval_questions.jsonl": "d39bf7b35b1d189ad717bb6f2f05b80303b601963816b09ca00f6da812105cdb",
    }
    assert {name: hashlib.sha256((core / name).read_bytes()).hexdigest() for name in expected} == expected


def test_core_v2_private_id_is_blocked_even_without_split_metadata(tmp_path):
    source = tmp_path / "verified.jsonl"
    source.write_text(json.dumps({
        "question_id": "CORE2-PE-MATH-001", "dataset_version": "core-v2",
        "question": "비공개 평가 질문", "final_answer": "노출 답",
        "verification_status": "passed", "training_approved": True,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(StudentDatasetError, match="CORE2-PE"):
        build_student_dataset(source, tmp_path / "student.jsonl", "S1")
