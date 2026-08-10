from __future__ import annotations

import json
from pathlib import Path

import pytest

from compass_ai.jsonl import read_jsonl, write_jsonl
from compass_ai.student_data import (
    StudentDatasetError,
    build_student_dataset,
    validate_student_dataset,
)
from compass_ai.student_training import (
    StudentConfigError,
    evaluate_saved_student,
    load_student_config,
    resolve_quantization_mode,
    run_student,
)


def _teacher_row(question_id: str = "train-001", teacher: str = "TeacherA", **overrides: object) -> dict:
    row = {
        "question_id": question_id,
        "question": f"{question_id}의 답을 구하라.",
        "final_answer": "최종 정답",
        "raw_response": {
            "choices": [{"message": {"reasoning_content": "숨겨야 할 추론", "content": "최종 정답"}}]
        },
        "teacher_model": teacher,
        "checkpoint": "teacher-checkpoint",
        "verification_status": "passed",
        "verification_outcome": "teacher_a_selected",
        "training_approved": True,
        "tag": "teacher_verified",
        "error_type": None,
        "evidence": "fixture evidence",
        "domain": "stem",
        "task_type": "calculation",
        "timestamp": "2026-08-10T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def _training_config(tmp_path: Path, *, condition: str = "S3b", training_enabled: bool = True) -> dict:
    return {
        "run_name": "student-test",
        "condition": condition,
        "base_model": str(tmp_path / "tiny-base"),
        "dataset": str(tmp_path / "student.jsonl") if training_enabled else None,
        "evaluation_dataset": str(tmp_path / "toy-eval.jsonl"),
        "output_dir": str(tmp_path / "run"),
        "training_enabled": training_enabled,
        "seed": 17,
        "epochs": 1,
        "learning_rate": 0.001,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_sequence_length": 64,
        "lora_rank": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["c_attn"],
        "fp16": False,
        "bf16": False,
        "quantization_mode": "none",
        "allow_quantization_fallback": True,
        "max_steps": 1,
    }


def test_student_dataset_builder_separates_final_answer_from_reasoning_and_records_checksum(tmp_path: Path):
    source = tmp_path / "verified.jsonl"
    output = tmp_path / "student.jsonl"
    write_jsonl(source, [_teacher_row(
        verification_method="answer_key",
        verification_result="verified",
        conflict_type="answer_conflict",
        teacher_a_raw_response_reference={"path": "a/responses.jsonl", "line": 1},
        teacher_b_raw_response_reference={"path": "b/responses.jsonl", "line": 1},
        verifier_metadata={"verifier": "independent", "version": 1},
        source_run_ids=["run-a", "run-b"],
    )])

    report = build_student_dataset(source, output, condition="S4_teacher_only")

    assert report["sample_count"] == 1
    assert len(report["dataset_checksum"]) == 64
    sample = read_jsonl(output)[0]
    assert sample["sample_id"].startswith("S4_teacher_only-")
    assert sample["source_question_id"] == "train-001"
    assert sample["prompt"] == "train-001의 답을 구하라."
    assert sample["target"] == "최종 정답"
    assert sample["source_teacher"] == "TeacherA"
    assert sample["verification_status"] == "passed"
    assert sample["condition"] == "S4_teacher_only"
    assert sample["provenance"]["conflict_type"] == "answer_conflict"
    assert sample["provenance"]["source_run_ids"] == ["run-a", "run-b"]
    serialized = json.dumps(sample, ensure_ascii=False)
    assert "숨겨야 할 추론" not in serialized
    assert "raw_response" not in sample


@pytest.mark.parametrize(
    "changes",
    [
        {"tag": "excluded"},
        {"verification_outcome": "excluded"},
        {"error_type": "format_error"},
        {"conflict_type": "answer_conflict"},
        {"response_status": "refusal"},
    ],
)
def test_student_dataset_builder_blocks_rejected_or_conflicting_samples(tmp_path: Path, changes: dict):
    source = tmp_path / "bad.jsonl"
    write_jsonl(source, [_teacher_row(**changes)])
    with pytest.raises(StudentDatasetError, match="rejected|conflict|error|response_status"):
        build_student_dataset(source, tmp_path / "student.jsonl", condition="S4_teacher_only")


def test_duplicate_question_is_blocked_except_distinct_teacher_samples_in_s3a(tmp_path: Path):
    source = tmp_path / "duplicate.jsonl"
    write_jsonl(source, [_teacher_row(teacher="TeacherA"), _teacher_row(teacher="TeacherB")])
    with pytest.raises(StudentDatasetError, match="duplicate source_question_id"):
        build_student_dataset(source, tmp_path / "s4.jsonl", condition="S4_teacher_only")

    report = build_student_dataset(source, tmp_path / "s3a.jsonl", condition="S3a")
    assert report["sample_count"] == 2
    assert len({row["sample_id"] for row in read_jsonl(tmp_path / "s3a.jsonl")}) == 2


def test_malformed_student_schema_and_pilot_source_are_rejected(tmp_path: Path):
    with pytest.raises(StudentDatasetError, match="missing"):
        validate_student_dataset([{"sample_id": "bad"}])

    source = tmp_path / "pilot.jsonl"
    write_jsonl(source, [_teacher_row(dataset_version="pilot-v1")])
    with pytest.raises(StudentDatasetError, match="Pilot-v1"):
        build_student_dataset(source, tmp_path / "student.jsonl", condition="S1")

    reference_source = tmp_path / "reference.jsonl"
    write_jsonl(reference_source, [_teacher_row(reference_answer="누출", rubric="R-LEAK")])
    with pytest.raises(StudentDatasetError, match="reference/rubric"):
        build_student_dataset(reference_source, tmp_path / "reference-student.jsonl", condition="S1")


def test_training_config_validation_and_quantization_fallback(tmp_path: Path):
    config_path = tmp_path / "student.yaml"
    config_path.write_text(json.dumps(_training_config(tmp_path), indent=2), encoding="utf-8")
    loaded = load_student_config(config_path)
    assert loaded["max_steps"] == 1

    missing = _training_config(tmp_path)
    missing.pop("lora_rank")
    config_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(StudentConfigError, match="lora_rank"):
        load_student_config(config_path)

    actual, reason = resolve_quantization_mode("4bit", allow_fallback=True, cuda_available=False, bitsandbytes_available=False)
    assert actual == "none"
    assert "fallback" in reason
    with pytest.raises(StudentConfigError, match="4bit"):
        resolve_quantization_mode("4bit", allow_fallback=False, cuda_available=False, bitsandbytes_available=False)


def test_s0_config_has_no_training_or_dataset(tmp_path: Path):
    config_path = tmp_path / "s0.yaml"
    config_path.write_text(json.dumps(_training_config(tmp_path, condition="S0", training_enabled=False)), encoding="utf-8")
    config = load_student_config(config_path)
    assert config["condition"] == "S0"
    assert config["training_enabled"] is False
    assert config["dataset"] is None


def test_training_refuses_nonempty_output_directory_before_loading_model(tmp_path: Path):
    (tmp_path / "tiny-base").mkdir()
    write_jsonl(
        tmp_path / "toy-eval.jsonl",
        [{"evaluation_id": "toy-eval-001", "prompt": "질문", "reference_answer": "답"}],
    )
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    config_path = tmp_path / "s0.yaml"
    config_path.write_text(
        json.dumps(_training_config(tmp_path, condition="S0", training_enabled=False)), encoding="utf-8"
    )
    with pytest.raises(StudentConfigError, match="output dir collision"):
        run_student(config_path, root=tmp_path)


def test_student_configs_preserve_existing_condition_meanings():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "s0.yaml": ("S0", False),
        "s1.yaml": ("S1", True),
        "s2.yaml": ("S2", True),
        "s3a.yaml": ("S3a", True),
        "s3b.yaml": ("S3b", True),
        "s3_matched.yaml": ("S3_matched", True),
        "s4.yaml": ("S4", True),
        "s4_teacher_only.yaml": ("S4_teacher_only", True),
        "s4_with_human_correction.yaml": ("S4_with_human_correction", True),
        "s5.yaml": ("S5", True),
    }
    for filename, (condition, enabled) in expected.items():
        config = load_student_config(root / "configs" / "student" / filename)
        assert (config["condition"], config["training_enabled"]) == (condition, enabled)


@pytest.mark.filterwarnings("error:fan_in_fan_out is set to False.*")
def test_tiny_transformers_lora_train_reload_inference_and_evaluation(tmp_path: Path, caplog):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from compass_ai.student_tiny import create_tiny_causal_lm
    from transformers import AutoConfig

    base = tmp_path / "tiny-base"
    create_tiny_causal_lm(base, seed=17)
    assert AutoConfig.from_pretrained(base, local_files_only=True).loss_type == "ForCausalLM"
    source = tmp_path / "teacher.jsonl"
    write_jsonl(source, [_teacher_row("train-001"), _teacher_row("train-002")])
    build_student_dataset(source, tmp_path / "student.jsonl", condition="S3b")
    write_jsonl(
        tmp_path / "toy-eval.jsonl",
        [{"evaluation_id": "toy-eval-001", "prompt": "테스트 질문", "reference_answer": "정답"}],
    )
    config = _training_config(tmp_path)
    config_path = tmp_path / "student.yaml"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    result = run_student(config_path, root=tmp_path)

    run_dir = Path(result["run_dir"])
    assert result["status"] == "complete"
    assert result["training_performed"] is True
    assert result["adapter_reload_verified"] is True
    assert result["dataset_checksum"] == build_student_dataset(
        source, tmp_path / "student-copy.jsonl", condition="S3b"
    )["dataset_checksum"]
    assert (run_dir / "adapter" / "adapter_config.json").exists()
    assert (run_dir / "trainer_state.json").exists()
    assert (run_dir / "config.snapshot.json").exists()
    assert (run_dir / "run_metadata.json").exists()
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["evaluation_count"] == 1
    assert evaluation["results"][0]["evaluation_id"] == "toy-eval-001"
    rerun = evaluate_saved_student(config_path, root=tmp_path)
    assert rerun["evaluation_count"] == 1
    assert (run_dir / "evaluation.rerun.json").exists()
    assert not any("loss_type=None" in record.message for record in caplog.records)


def test_s0_uses_same_evaluator_without_training(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from compass_ai.student_tiny import create_tiny_causal_lm

    create_tiny_causal_lm(tmp_path / "tiny-base", seed=17)
    write_jsonl(
        tmp_path / "toy-eval.jsonl",
        [{"evaluation_id": "toy-eval-001", "prompt": "기준선 질문", "reference_answer": "정답"}],
    )
    config = _training_config(tmp_path, condition="S0", training_enabled=False)
    config_path = tmp_path / "s0.yaml"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    result = run_student(config_path, root=tmp_path)

    assert result["training_performed"] is False
    assert result["adapter_path"] is None
    assert result["dataset_checksum"] is None
    assert Path(result["run_dir"], "evaluation.json").exists()
